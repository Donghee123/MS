import os
import csv
import random

import torch as T
import torch.nn.functional as F
from torch.optim import Adam

import numpy as np
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork, ValueNetwork
from dqnagent import Agent

import matplotlib.pyplot as plt

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
    
class Agent():
    def __init__(self, dqnagent, action_space, env, args, alpha=0.0003, beta=0.0003, input_dims=[83],
            gamma=0.99, max_size=1000000, tau=0.005,
            layer1_size=512, layer2_size=256, batch_size=256, reward_scale=2):
        
        self.train_graph_step= args.train_graph_step
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.dqnagent = dqnagent
        self.args = args
        self.test_step = args.test_step
        n_actions = action_space.shape[0]
        self.step = 0
        self.train_step = args.train_step
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = action_space.shape[0]
        self.RB_number = 20
        self.num_vehicle = env.n_Veh
        self.action_all_with_power = np.zeros([self.num_vehicle, 3, 2],dtype = 'float')   # this is actions that taken by V2V links with power
        self.action_all_with_power_training = np.zeros([20, 3, 2],dtype = 'float')   # this is actions that taken by V2V links with power
        
        self.actor = ActorNetwork(alpha, input_dims, n_actions=n_actions,
                    name='actor', max_action=action_space.high)
        self.critic_1 = CriticNetwork(beta, input_dims, n_actions=n_actions,
                    name='critic_1')
        self.critic_2 = CriticNetwork(beta, input_dims, n_actions=n_actions,
                    name='critic_2')
        self.value = ValueNetwork(beta, input_dims, name='value')
        self.target_value = ValueNetwork(beta, input_dims, name='target_value')

        self.scale = reward_scale
        self.update_network_parameters(tau=1)
        

    def select_action(self, state):
        state = T.Tensor([state]).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)

        return actions.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() + \
                    (1-tau)*target_value_state_dict[name].clone()

        self.target_value.load_state_dict(value_state_dict)

    def save_models(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        
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
    
    
    """
    Clip 기능 추가 2021/10/31
    """
    
    def ClipAction(self, action : np.array):              
        selected_powerdB = action
        
        if selected_powerdB < 0.0:
            selected_powerdB = 0.0
        elif selected_powerdB > 23.0:
            selected_powerdB = 23.0
        
        return selected_powerdB
    
    def merge_action(self, idx, action):
        select_ResourceBlock = action[0]
        select_PowerdBm = self.ClipAction(action[1]) 
        self.action_all_with_power[idx[0], idx[1], 0] = int(select_ResourceBlock)
        self.action_all_with_power[idx[0], idx[1], 1] = select_PowerdBm 
    
    def train_with_dqn(self): 
        
        random_choice = False      
        updates = 0
        total_numsteps = 0
        self.env.new_random_game(20)      
        rewardloggingDataHeader = ['reward']
        
        rewardloggingData = []
        
        
        for self.step in (range(0, self.train_step)): # need more configuration #40000
             
            #for train show
            train_selectPowerList = []  
                                   
            if (self.step % 2000 == 1):
                self.env.new_random_game(20)
                
            print(self.step)
            self.training = True
            reward_sum = 0
            
            
            #현재 상태에서 모든 차량이 선택을 수행함.
            for k in range(1):
                #i번째 송신 차량
                for i in range(len(self.env.vehicles)):
                    #i번째 송신 차량에서 전송한 신호를 수신하는 j번째 수신 차량 
                    for j in range(3): 
                        
                        # i번째 차량에서 j번째 차량으로 데이터를 전송할때 state를 가져옴.
                        state_old = self.get_state([i,j]) 
                        
                        #dqn agent가 resource block을 우선 선택함.
                        action = self.dqnagent.predict(state_old, 0, True, random_choice = random_choice)
                        
                        #self.merge_action([i,j], action)   
                        selectedResourceblock = action % self.RB_number 
                        
                        state_old = list(state_old)
                        state_old.append(selectedResourceblock)
                        state_old = np.array(state_old)
                        
                        #state를 보고 action을 정함
                        #action은 선택한 power level, 선택한 resource block 정보를 가짐
                        # 랜덤 선택
                        if self.args.start_steps > total_numsteps:                            
                            powerdBm = random.uniform(0.0, 23.0)
                            action = np.array([powerdBm])
                        else:
                            action = self.select_action(state_old)
                            #print('selcted greedy action : ', action)
                        
                        train_selectPowerList.append(action[0])
                        
                        #print('select RB : ',selectedResourceblock, ' PowerdBm : ', action[0])
                        
                        if total_numsteps % 50 == 0:
                            # 업데이트 
                            self.learn()

                        total_numsteps+=1
                        #self.merge_action([i,j], action)   
                        
                        #선택 selected_resourceBlock, selected_powerdB                 
                        selected_powerdB = self.ClipAction(action)
                         
                        #i 번째 차량에서 j 번째 차량으로 전송할 리소스 블럭 선택
                        self.action_all_with_power_training[i, j, 0] = selectedResourceblock  # 선택한 Resourceblock을 저장함. 
    
                        #i 번째 차량에서 j 번째 차량으로 전송할 Power dB 선택
                        self.action_all_with_power_training[i, j, 1] = selected_powerdB # PowerdBm을 저장함.
                        
                        #print(self.action_all_with_power_training)
                        
                        #선택한 power level과 resource block을 기반으로 reward를 계산함.
                        reward_train = self.env.act_for_training(self.action_all_with_power_training, [i,j]) 
                        
                        reward_sum += reward_train
                        
                        state_new = self.get_state([i,j]) 
                        
                        #dqn agent가 resource block을 우선 선택함.
                        next_action = self.dqnagent.predict(state_new, 0, True, random_choice = random_choice)
                        
                        #self.merge_action([i,j], action)   
                        next_selectedResourceblock = next_action % self.RB_number 
                        
                        state_new = list(state_new)
                        state_new.append(next_selectedResourceblock)
                        state_new = np.array(state_new)                 
                        #self.observe(state_old, state_new, reward_train, action)
                        self.remember(state_old.reshape(83), action, np.array([reward_train]), state_new.reshape(83),np.array([0])) # Append transition to memory
                    
                    
            
            
            if self.step % self.train_graph_step == 0:
                plt.hist(train_selectPowerList, bins=230, density=True, alpha=0.7, histtype='stepfilled')                       
                plt.title('train')
                plt.show()
                        
            print(reward_sum)                                                      
            rewardloggingData.append(reward_sum/60.0)
            
            if (self.step % self.test_step == 0) and (self.step > 0):
                # testing 
                self.training = False
                number_of_game = 10
                if (self.step % 10000 == 0) and (self.step > 0):
                    number_of_game = 50 
                if (self.step == 38000):
                    number_of_game = 100               
                V2I_Rate_list = np.zeros(number_of_game)
                V2V_Rate_list = np.zeros(number_of_game)
                Fail_percent_list = np.zeros(number_of_game)
                
                #for show
                selectPowerList = []

                
                for game_idx in range(number_of_game):
                    self.env.new_random_game(self.num_vehicle)
                    test_sample = 200
                    V2IRate_list = []
                    V2VRate_list = []
                    print('test game idx:', game_idx)
                    for k in range(test_sample):
                        action_temp = self.action_all_with_power.copy()
                        for i in range(len(self.env.vehicles)):
                            self.action_all_with_power[i,:,0] = -1
                            sorted_idx = np.argsort(self.env.individual_time_limit[i,:])          
                            for j in sorted_idx:                   
                                state_old = self.get_state([i,j])
                                selectedRB = self.dqnagent.predict(state_old, 0, True, random_choice = random_choice)
                                selectedRB = selectedRB % self.RB_number 
                                state_old = list(state_old)
                                state_old.append(selectedRB)
                                
                                state_old = np.array(state_old)                               
                                selectedPowerdBm = self.select_action(state_old)                               
                                selectedPowerdBm = self.ClipAction(selectedPowerdBm)
                                selectPowerList.append(selectedPowerdBm)
                                #print(action)
                                action = np.array([selectedRB, selectedPowerdBm.item()])                                
                                self.merge_action([i,j], action)
                            if i % (len(self.env.vehicles)/10) == 1:
                                action_temp = self.action_all_with_power.copy()
                                V2IRate, V2VRate, percent = self.env.act_asyn(action_temp) #self.action_all)            
                                V2IRate_list.append(np.sum(V2IRate))
                                V2VRate_list.append(np.sum(V2VRate))
                        #print("actions", self.action_all_with_power)
                    V2I_Rate_list[game_idx] = np.mean(np.asarray(V2IRate_list))
                    V2V_Rate_list[game_idx] = np.mean(np.asarray(V2VRate_list))
                    Fail_percent_list[game_idx] = percent
                    #print("action is", self.action_all_with_power)
                    print('failure probability is, ', percent)
                    
                    #plt.hist(selectPowerList, bins=230, density=True, alpha=0.7, histtype='stepfilled')
                    #plt.title('play')
                    #plt.show()
                    #print('action is that', action_temp[0,:])
                    
                    
                    
                self.save_models()
                
                #2000번 트레이닝 할때 마다 모델 저장.
                #self.save_model('V2X_Model_' + str(self.step) + '_' + str(np.mean(V2I_Rate_list) + np.mean(V2V_Rate_list)) + '_' + str(np.mean(Fail_percent_list)))
                print ('The number of vehicle is ', len(self.env.vehicles))
                print ('Mean of the V2I rate + V2V rate is that ', np.mean(V2I_Rate_list) + np.mean(V2V_Rate_list))
                print ('Mean of the V2I rate is that ', np.mean(V2I_Rate_list))
                print ('Mean of the V2V rate is that ', np.mean(V2V_Rate_list))
                print('Mean of Fail percent is that ', np.mean(Fail_percent_list))    
                                
    
        allTrainRewardLogingData=[]
        allTrainRewardLogingData.append(np.array(rewardloggingData))
        allTrainRewardLogingData = np.transpose(allTrainRewardLogingData)
            
        folderPath = './ResultTrainData'
        csvFileName = 'ResultTrainData.csv'
        csvrewardFileName = 'ResultRewardTrainData.csv'
            
        createFolder(folderPath)
        MakeCSVFile(folderPath, csvrewardFileName, rewardloggingDataHeader, allTrainRewardLogingData)
        self.save_models()
        #self.save_model('V2X_Model')
        
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = \
                self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.actor.device)
        state = T.tensor(state, dtype=T.float).to(self.actor.device)
        action = T.tensor(action, dtype=T.float).to(self.actor.device)

        value = self.value(state).view(-1)
        value_ = self.target_value(state_).view(-1)
        value_[done] = 0.0

        actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)
        
        actor_loss = log_probs - critic_value
        actor_loss = T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.scale*reward + self.gamma*value_
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.update_network_parameters()

