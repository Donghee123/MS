import random
from Environment import *
import wandb 
import argparse
from ppo import PPO
import torch 

sumrateV2IList = []
sumrateV2VList = []

probabilityOfSatisfiedV2VList = []

def find_nearest_value(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]    
    
def find_nearest_arg(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

def GetRB_Power(powerMin,action, env):
    
    powerRange = 23.0 - powerMin
    
    select_maxValue  = np.max(action[0:20])
    candidateList_selRB = np.where(action[0:20] == select_maxValue)
    selectedRB_index = random.sample(list(candidateList_selRB[0]),k=1)[0]

    
    actionFromPolicy = (action[20] + 1.0) * (powerRange / 2) + powerMin 
    selectedPower_index = find_nearest_arg(env.V2V_power_dB_List, actionFromPolicy)
     
    return selectedRB_index, selectedPower_index

def train(env, agent, train_iter, power_min, wandb):

    ep_reward = 0, 0, 0.
    ep_reward = [], []       
    power_min = power_min
    env.new_random_game(20)
    
    action_std_decay_rate = 0.000001
    min_action_std = 0.00001

    update_timestep = 10
    logs = []
    for step in (range(0, train_iter)): # need more configuration

        if step == 0:                   # initialize set some varibles
            num_game, update_count,ep_reward, ep_target_reward = 0., 0., 0., 0.
            total_reward, total_loss, total_q = 0., 0., 0.
            actions = []        
        
        #모든 차량이 2000번 action을 완료했을 경우 초기화
        if (step % 2000 == 1):
            env.new_random_game(20)
  
        print(step)
        state_old = env.get_state([0,0], True, env.action_all_with_power_training, env.action_all_with_power) 
        
        training = True
    

        #i번째 송신 차량
        for i in range(len(env.vehicles)):
            #i번째 송신 차량에서 전송한 신호를 수신하는 j번째 수신 차량 
            for j in range(3): 
                # i번째 차량에서 j번째 차량으로 데이터를 전송할때 state를 가져옴.
                 

                state_old = env.get_state([i,j], True, env.action_all_with_power_training, env.action_all_with_power) 
                                
                #state를 보고 action을 정함
                #action은 선택한 power level, 선택한 resource block 정보를 가짐
                action = agent.select_action(state_old)

                selectedRB_Index, selectedPower_Index = GetRB_Power(power_min, action, env)

                    #선택한 resource block을 넣어줌
                env.action_all_with_power_training[i, j, 0] = selectedRB_Index
                    
                    #선택한 power level을 넣어줌
                env.action_all_with_power_training[i, j, 1] = selectedPower_Index 
                                     
                    #선택한 power level과 resource block을 기반으로 reward를 계산함.
                reward_train, reward_best = env.act_for_training(env.action_all_with_power_training, [i,j])
                    
                ep_reward = ep_reward + reward_train
                ep_target_reward = ep_target_reward + reward_best
                    
                state_new = env.get_state([i,j], True, env.action_all_with_power_training, env.action_all_with_power) 
                    
                    #경험 저장
                is_terminal = 0
                agent.buffer.rewards.append(reward_train)
                agent.buffer.is_terminals.append(is_terminal)

        # update PPO agent
        if step % update_timestep == 0 and step > 3:
            agent.update()

        # if continuous action space; then decay action std of ouput action distribution
        if agent.has_continuous_action_space and step % update_timestep == 0:
            agent.decay_action_std(action_std_decay_rate, min_action_std)
        
        logs.append((": 2000 step Cumulative Reward : " + str(ep_reward) + '-- Target Reward : ' + str(ep_target_reward) + '-- Diff PPO vs target : ' + str(ep_target_reward - ep_reward)))
        wandb.log({"PPO_reward": ep_reward, "target_reward" : ep_target_reward, "Diff PPO vs target" : ep_target_reward - ep_reward})
        ep_reward = 0.
        ep_target_reward = 0.
                             
        if (step % 2000 == 0) and (step > 0):
            # testing 
            training = False
            number_of_game = 10
            if (step % 10000 == 0) and (step > 0):
                number_of_game = 50 
            if (step == 38000):
                number_of_game = 100               
            V2I_V2X_Rate_list = np.zeros(number_of_game)
            V2I_Rate_list = np.zeros(number_of_game)
            V2V_Rate_list = np.zeros(number_of_game)
            Fail_percent_list = np.zeros(number_of_game)
            resourceBlocks = [[0] for _ in range(20)]
            for game_idx in range(number_of_game):
                env.new_random_game(20)
                test_sample = 200
                Rate_list = []
                temp_V2V_Rate_list = []
                temp_V2I_Rate_list = []
                print('test game idx:', game_idx)
                for k in range(test_sample):
                    action_temp = env.action_all_with_power.copy()
                    for i in range(len(env.vehicles)):
                        env.action_all_with_power[i,:,0] = -1
                        sorted_idx = np.argsort(env.individual_time_limit[i,:])          
                        for j in sorted_idx:                   
                            state_old = env.get_state([i,j], False, env.action_all_with_power_training, env.action_all_with_power) 

                            action = agent.select_action(state_old)

                            selectedRB_Index, selectedPower_Index = GetRB_Power(power_min, action, env)
                            
                            env.action_all_with_power[i, j, 0] = selectedRB_Index
                            env.action_all_with_power[i, j, 1] = selectedPower_Index
                            
                            
                        if i % (len(env.vehicles)/10) == 1:
                            action_temp = env.action_all_with_power.copy()                                
                            returnV2IReward, returnV2VReward, percent = env.act_asyn(action_temp) #self.action_all)            
                            Rate_list.append(np.sum(returnV2IReward) + np.sum(returnV2VReward))
                            temp_V2I_Rate_list.append(np.sum(returnV2IReward))
                            temp_V2V_Rate_list.append(np.sum(returnV2VReward))
                            
                V2I_V2X_Rate_list[game_idx] = np.mean(np.asarray(Rate_list))
                V2I_Rate_list[game_idx] = np.mean(np.asarray(temp_V2I_Rate_list))
                V2V_Rate_list[game_idx] = np.mean(np.asarray(temp_V2V_Rate_list))
                Fail_percent_list[game_idx] = percent
                print('failure probability is, ', percent)
            
            histTable = wandb.Table(data=resourceBlocks, columns=["Resource block"])
            wandb.log({'my_histogram': wandb.plot.histogram(histTable, "Resource block",title="Count of Resource block")})   
            print ('The number of vehicle is ', len(env.vehicles))
            print ('Mean of the V2I + V2I rate is that ', np.mean(V2I_V2X_Rate_list))
            print ('Mean of the V2I rate is that ', np.mean(V2I_Rate_list))
            print ('Mean of the V2V rate is that ', np.mean(V2V_Rate_list))
            print('Mean of Fail percent is that ', np.mean(Fail_percent_list))                   
      
def main(args):

  wandb.init(config=args, project="my-project")
  wandb.config["My pytorch PPO"] = "PPO Version 0.1"

  up_lanes = [3.5/2,3.5/2 + 3.5,250+3.5/2, 250+3.5+3.5/2, 500+3.5/2, 500+3.5+3.5/2]
  down_lanes = [250-3.5-3.5/2,250-3.5/2,500-3.5-3.5/2,500-3.5/2,750-3.5-3.5/2,750-3.5/2]
  left_lanes = [3.5/2,3.5/2 + 3.5,433+3.5/2, 433+3.5+3.5/2, 866+3.5/2, 866+3.5+3.5/2]
  right_lanes = [433-3.5-3.5/2,433-3.5/2,866-3.5-3.5/2,866-3.5/2,1299-3.5-3.5/2,1299-3.5/2]

  arrayOfVeh = [20] # for train
  
  width = 750
  height = 1299

  power_min = -10

  Env = Environ(down_lanes,up_lanes,left_lanes,right_lanes, width, height, arrayOfVeh[0], power_min)
  
  #모델 정의
  n_input = 82
  n_output = 21
  has_continuous_action_space = True
  action_std = 0.7

  K_epochs = 10               # update policy for K epochs
  eps_clip = 0.2              # clip parameter for PPO
  gamma = 0.99                # discount factor

  lr_actor = 0.0003       # learning rate for actor network
  lr_critic = 0.001       # learning rate for critic network

  ppo_agent = PPO(n_input, n_output, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

  train_iter = args.train_iter
  
  for nVeh in arrayOfVeh:      
      
      Env.new_random_game()    
      #학습
      train(env=Env, agent=ppo_agent, train_iter=train_iter, power_min=power_min, wandb=wandb)

  
  
if __name__ == '__main__':
    

    parser = argparse.ArgumentParser(description='PPO with Pytorch')

    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
    parser.add_argument('--actor_learning_rate', default=0.0001, type=float, help='actor learning rate')   
    parser.add_argument('--critic_learning_rate', default=0.0001, type=float, help='critic learning rate')   
    parser.add_argument('--train_iter', default=40000, type=int, help='train iters each timestep')

    args = parser.parse_args()
    print('test will be cuda :', torch.cuda.is_available())
    main(args)
