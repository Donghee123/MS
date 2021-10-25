import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update
from model import GaussianPolicy, QNetwork, DeterministicPolicy
import numpy as np
from replay_memory import ReplayMemory
import random
import pandas as pd
import csv
import os

#File 유틸 함수들    
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
     
def MakeCSVFile(strFolderPath, strFilePath, aryOfHedaers, aryOfDatas):
    strTotalPath = "%s\%s" % (strFolderPath,strFilePath)
    
    f = open(strTotalPath,'w', newline='')
    wr = csv.writer(f)
    wr.writerow(aryOfHedaers)
    
    for i in range(0,len(aryOfDatas)):
        wr.writerow(aryOfDatas[i])
    
    f.close()
    
class SAC(object):
    def __init__(self, num_inputs, action_space, args, env):

        self.env = env
        self.num_vehicle = env.n_Veh
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.action_space = action_space
        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning
        self.args = args
        self.action_all_with_power = np.zeros([self.num_vehicle, 3, 2],dtype = 'float')   # this is actions that taken by V2V links with power
        self.action_all_with_power_training = np.zeros([20, 3, 2],dtype = 'float')   # this is actions that taken by V2V links with power
        self.memory = ReplayMemory(args.replay_size, args.seed)
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.RB_number = 20
        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)
        self.training = True
        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape[0]).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return float(action.detach().cpu()[0])

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/sac_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/sac_critic_{}_{}".format(env_name, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    # Load model parameters
    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))
    
    def get_state(self, idx):
    # ===============
    #  Get State from the environment
    # =============
        vehicle_number = len(self.env.vehicles)
        
        #idx번째 차량이 전송하고자하는 v2v link의 resource block의 채널 상태를 보여줌
        V2V_channel = (self.env.V2V_channels_with_fastfading[idx[0],self.env.vehicles[idx[0]].destinations[idx[1]],:] - 80)/60
        
        #idx번째 차량이 전송하고자하는 v2i link의 resource block의 채널 상태를 보여줌
        V2I_channel = (self.env.V2I_channels_with_fastfading[idx[0], :] - 80)/60
        
        #이전 스탭에서 idx번째 차량이 전송하고자하는 v2v link의 resource block에서 살펴 볼 수 있는 Interference
        V2V_interference = (-self.env.V2V_Interference_all[idx[0],idx[1],:] - 60)/60
        #선택한 resource block
        NeiSelection = np.zeros(self.RB_number)
        
        #인접한 차량에게 전송할 power 선정
        for i in range(3):
            for j in range(3):
                if self.training:
                    NeiSelection[int(self.action_all_with_power_training[self.env.vehicles[idx[0]].neighbors[i], j, 0 ])] = 1
                else:
                    NeiSelection[int(self.action_all_with_power[self.env.vehicles[idx[0]].neighbors[i], j, 0 ])] = 1
                   
        for i in range(3):
            if i == idx[1]:
                continue
            if self.training:
                if self.action_all_with_power_training[idx[0],i,0] >= 0:
                    NeiSelection[int(self.action_all_with_power_training[idx[0],i,0])] = 1
            else:
                if self.action_all_with_power[idx[0],i,0] >= 0:
                    NeiSelection[int(self.action_all_with_power[idx[0],i,0])] = 1
                    
        time_remaining = np.asarray([self.env.demand[idx[0],idx[1]] / self.env.demand_amount])
        load_remaining = np.asarray([self.env.individual_time_limit[idx[0],idx[1]] / self.env.V2V_limit])
        #print('shapes', time_remaining.shape,load_remaining.shape)
        # V2I_channel : #idx번째 차량이 전송하고자하는 v2i link의 resource block의 채널 상태를 보여줌
        # V2V_interference : 이전 스탭에서 idx번째 차량이 전송하고자하는 v2v link의 resource block에서 볼 수 있는 Interference
        # V2V_channel : #idx번째 차량이 전송하고자하는 v2v link의 resource block의 채널 상태를 보여줌
        # 근접한 차량이 선택한 리소스 블록 상태
        # 남은 시간
        # 걸린 시간
        return np.concatenate((V2I_channel, V2V_interference, V2V_channel, NeiSelection, time_remaining, load_remaining))#,time_remaining))
        #return np.concatenate((V2I_channel, V2V_interference, V2V_channel, time_remaining, load_remaining))#,time_remaining))
    
    """
    Scaling 기능 추가 2021/10/25
    0~999 -> -100.0 ~ 123.0
    """
    def GenerateAction(self, action : float):
        
        selected_resourceBlock = 0
        selected_powerdB = -100.0
        
        onestep_PowerdB = 123.0/999.0
        
        if action < 0.0:
            selected_resourceBlock = 0
            selected_powerdB = 0.0
        elif action >= 20000:
            selected_resourceBlock = 19
            selected_powerdB = 123.0
        else:
            selected_resourceBlock = int(action / 1000)
            selected_powerdB = (action % 1000) * onestep_PowerdB
                            
        if selected_powerdB > 123.0:
            selected_powerdB = 123.0
                            
        selected_powerdB -= 100.0
        
        return selected_resourceBlock, selected_powerdB
    
    def merge_action(self, idx, action):
        select_ResourceBlock, select_PowerdB = self.GenerateAction(action)
        self.action_all_with_power[idx[0], idx[1], 0] = select_ResourceBlock
        self.action_all_with_power[idx[0], idx[1], 1] = select_PowerdB 
        
    def play(self, actor_path, critic_path , n_step = 100, n_episode = 100, test_ep = None):
        
        self.load_model(actor_path = actor_path, critic_path = critic_path)
        number_of_game = n_episode
        V2I_Rate_list = np.zeros(number_of_game)
        V2V_Rate_list = np.zeros(number_of_game)
        Fail_percent_list = np.zeros(number_of_game)  
        self.training = False


        for game_idx in range(number_of_game):
            self.env.new_random_game(self.num_vehicle)
            test_sample = n_step
            Rate_list = []
            Rate_list_V2V = []
            
            print('test game idx:', game_idx)
            print('The number of vehicle is ', len(self.env.vehicles))
            
            time_left_list = []


            for k in range(test_sample):
                action_temp = self.action_all_with_power.copy()
                for i in range(len(self.env.vehicles)):
                    self.action_all_with_power[i, :, 0] = -1
                    sorted_idx = np.argsort(self.env.individual_time_limit[i, :])
                    for j in sorted_idx:
                        state_old = self.get_state([i, j])
                        time_left_list.append(state_old[-1])
                        action = self.select_action(state_old, evaluate=True)
                        
                        if state_old[-1] <=0:
                            continue
                        
                        selectResouceBlock, selectPowerDB = self.GenerateAction(action)                                            
                        
                        self.merge_action([i, j], action)
                    
                    #시뮬레이션 차량의 갯수 / 10 만큼 action이 정해지면 act를 수행함.
                    if i % (len(self.env.vehicles) / 10) == 1:
                        action_temp = self.action_all_with_power.copy()
                        rewardOfV2I, rewardOfV2V, percent = self.env.act_asyn(action_temp)  # self.action_all)
                        Rate_list.append(np.sum(rewardOfV2I))
                        Rate_list_V2V.append(np.sum(rewardOfV2V))
                        
                # print("actions", self.action_all_with_power)
            
            
          
            V2I_Rate_list[game_idx] = np.mean(np.asarray(Rate_list))
            V2V_Rate_list[game_idx] = np.mean(np.asarray(Rate_list_V2V))
            
            Fail_percent_list[game_idx] = percent

            print('Mean of the V2I rate is that ', np.mean(V2I_Rate_list[0:game_idx] ))
            print('Mean of the V2V rate is that ', np.mean(V2V_Rate_list[0:game_idx] ))
            print('Mean of Fail percent is that ',percent, np.mean(Fail_percent_list[0:game_idx]))
            # print('action is that', action_temp[0,:])

        print('The number of vehicle is ', len(self.env.vehicles))
        print('Mean of the V2I rate is that ', np.mean(V2I_Rate_list))
        print('Mean of the V2V rate is that ', np.mean(V2V_Rate_list))
        print('Mean of Fail percent is that ', np.mean(Fail_percent_list))
        # print('Test Reward is ', np.mean(test_result))
        
        return np.mean(V2I_Rate_list), np.mean(V2V_Rate_list),np.mean(Fail_percent_list)
    def train(self): 
        
        updateloggingData = []
        updateloggingDataHeader = ['critic_1_loss', 'critic_2_loss', 'policy_loss', 'ent_loss', 'alpha']
        
        trainloggingData = []
        trainloggingDataHeader = ['reward']
        
        updates = 0
        num_game, self.update_count, ep_reward = 0, 0, 0.
        total_reward, self.total_loss, self.total_q = 0.,0.,0.
        max_avg_ep_reward = 0
        ep_reward, actions = [], []        
        mean_big = 0
        number_big = 0
        mean_not_big = 0
        number_not_big = 0
        total_numsteps = 0
        self.env.new_random_game(20)
        
        updateloggingDataHeader = ['critic_1_loss', 'critic_2_loss', 'policy_loss', 'ent_loss', 'alpha'] 
        rewardloggingDataHeader = ['reward']
        
        critic_1_losses = []
        critic_2_losses = []
        policy_losses = []
        ent_losses = []
        alphas = []
        rewardloggingData = []
        
        
        
        for self.step in (range(0, 50000)): # need more configuration #50000
            if self.step == 0:                   # initialize set some varibles
                num_game, self.update_count,ep_reward = 0, 0, 0.
                total_reward, self.total_loss, self.total_q = 0., 0., 0.
                ep_reward, actions = [], []               
                
            # prediction
            # action = self.predict(self.history.get())
            if (self.step % 2000 == 1):
                self.env.new_random_game(20)
                
            print(self.step)
            state_old = self.get_state([0,0])
            #print("state", state_old)
            self.training = True
            reward_sum = 0
            
            temp_critic_1_losses = []
            temp_critic_2_losses = []
            temp_policy_losses = []
            temp_ent_losses = []
            temp_alphas = []
            
            for k in range(1):
                #i번째 송신 차량
                for i in range(len(self.env.vehicles)):
                    #i번째 송신 차량에서 전송한 신호를 수신하는 j번째 수신 차량 
                    for j in range(3): 
                        # i번째 차량에서 j번째 차량으로 데이터를 전송할때 state를 가져옴.
                        state_old = self.get_state([i,j]) 
                        
                        #state를 보고 action을 정함
                        #action은 선택한 power level, 선택한 resource block 정보를 가짐
                        # 랜덤 선택
                        if self.args.start_steps > total_numsteps:
                            resourceblock = random.randint(0, 19)
                            powerdBm = random.uniform(0.0, 123.0)
                            action = (resourceblock * 1000) + powerdBm
                        else:
                            action = self.select_action(state_old)
                            print('selcted greedy action : ', action)
                        
                        # 업데이트 
                        if len(self.memory) > self.args.batch_size:
                            # Number of updates per step in environment
                            for z in range(self.args.updates_per_step):
                                # Update parameters of all the networks
                                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = self.update_parameters(self.memory, self.args.batch_size, updates)
                                updates += 1
                                #print('parameter update count : ', updates)                              
                                temp_critic_1_losses.append(critic_1_loss)
                                temp_critic_2_losses.append(critic_2_loss)
                                temp_policy_losses.append(policy_loss)
                                temp_ent_losses.append(ent_loss)
                                temp_alphas.append(alpha)
                          

                        total_numsteps+=1
                        #self.merge_action([i,j], action)   
                        
                        #선택 selected_resourceBlock, selected_powerdB                 
                        selected_resourceBlock , selected_powerdB = self.GenerateAction(action)
                         
                        #i 번째 차량에서 j 번째 차량으로 전송할 리소스 블럭 선택
                        self.action_all_with_power_training[i, j, 0] = int(selected_resourceBlock)  # Resourceblock 인덱스 
                        
                        #i 번째 차량에서 j 번째 차량으로 전송할 Power dB 선택
                        self.action_all_with_power_training[i, j, 1] = selected_powerdB #-100~23 power dB
                                         
                        #선택한 power level과 resource block을 기반으로 reward를 계산함.
                        reward_train = self.env.act_for_training(self.action_all_with_power_training, [i,j]) 
                        
                        reward_sum += reward_train
                        
                        state_new = self.get_state([i,j]) 
                        
                        #self.observe(state_old, state_new, reward_train, action)
                        
                        self.memory.push(state_old.reshape(82), np.array([action]), np.array([reward_train]), state_new.reshape(82),np.array([1])) # Append transition to memory
            
            critic_1_losses.append(np.mean(temp_critic_1_losses))
            critic_2_losses.append( np.mean(temp_critic_2_losses))
            policy_losses.append(np.mean(temp_policy_losses))
            ent_losses.append(np.mean(temp_ent_losses))
            alphas.append(np.mean(temp_alphas)   )                                                         
            rewardloggingData.append(reward_sum/60.0)
            
            """
            if (self.step % 2000 == 0) and (self.step > 0):
                # testing 
                self.training = False
                number_of_game = 10
                if (self.step % 10000 == 0) and (self.step > 0):
                    number_of_game = 50 
                if (self.step == 38000):
                    number_of_game = 100               
                V2I_Rate_list = np.zeros(number_of_game)
                Fail_percent_list = np.zeros(number_of_game)
                for game_idx in range(number_of_game):
                    self.env.new_random_game(self.num_vehicle)
                    test_sample = 200
                    Rate_list = []
                    print('test game idx:', game_idx)
                    for k in range(test_sample):
                        action_temp = self.action_all_with_power.copy()
                        for i in range(len(self.env.vehicles)):
                            self.action_all_with_power[i,:,0] = -1
                            sorted_idx = np.argsort(self.env.individual_time_limit[i,:])          
                            for j in sorted_idx:                   
                                state_old = self.get_state([i,j])
                                action = self.select_action(state_old)
                                self.merge_action([i,j], action)
                            if i % (len(self.env.vehicles)/10) == 1:
                                action_temp = self.action_all_with_power.copy()
                                reward, percent = self.env.act_asyn(action_temp) #self.action_all)            
                                Rate_list.append(np.sum(reward))
                        #print("actions", self.action_all_with_power)
                    V2I_Rate_list[game_idx] = np.mean(np.asarray(Rate_list))
                    Fail_percent_list[game_idx] = percent
                    #print("action is", self.action_all_with_power)
                    print('failure probability is, ', percent)
                    #print('action is that', action_temp[0,:])
                    
                self.save_model('V2X_Model')
                print ('The number of vehicle is ', len(self.env.vehicles))
                print ('Mean of the V2I rate is that ', np.mean(V2I_Rate_list))
                print('Mean of Fail percent is that ', np.mean(Fail_percent_list))                   
                #print('Test Reward is ', np.mean(test_result))
              
            """    
             
                                
        allTrainLogingData=[]
        allTrainLogingData.append(np.array(critic_1_losses))
        allTrainLogingData.append(np.array(critic_2_losses))
        allTrainLogingData.append(np.array(policy_losses))
        allTrainLogingData.append(np.array(ent_losses))
        allTrainLogingData = np.transpose(allTrainLogingData)
  
        allTrainRewardLogingData=[]
        allTrainRewardLogingData.append(np.array(rewardloggingData))
        allTrainRewardLogingData = np.transpose(allTrainRewardLogingData)
            
        folderPath = './ResultTrainData'
        csvFileName = 'ResultTrainData.csv'
        csvrewardFileName = 'ResultRewardTrainData.csv'
            
        createFolder(folderPath)
        MakeCSVFile(folderPath, csvFileName, updateloggingDataHeader, allTrainLogingData)
        MakeCSVFile(folderPath, csvrewardFileName, rewardloggingDataHeader, allTrainRewardLogingData)
        self.save_model('V2X_Model')
