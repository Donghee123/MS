from __future__ import division, print_function
import random
import scipy
import scipy.io
import numpy as np
import Environment_marl
import os
import sys
import argparse

import torch
import numpy as np
from sac_torch import Agent
import pandas as pd
import csv

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
    

# ################## SETTINGS ######################
up_lanes = [i/2.0 for i in [3.5/2,3.5/2 + 3.5,250+3.5/2, 250+3.5+3.5/2, 500+3.5/2, 500+3.5+3.5/2]]
down_lanes = [i/2.0 for i in [250-3.5-3.5/2,250-3.5/2,500-3.5-3.5/2,500-3.5/2,750-3.5-3.5/2,750-3.5/2]]
left_lanes = [i/2.0 for i in [3.5/2,3.5/2 + 3.5,433+3.5/2, 433+3.5+3.5/2, 866+3.5/2, 866+3.5+3.5/2]]
right_lanes = [i/2.0 for i in [433-3.5-3.5/2,433-3.5/2,866-3.5-3.5/2,866-3.5/2,1299-3.5-3.5/2,1299-3.5/2]]

width = 750/2
height = 1298/2

IS_TRAIN = 1
IS_TEST = 1-IS_TRAIN

n_veh = 4
n_neighbor = 1
n_RB = n_veh

env = Environment_marl.Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height, n_veh, n_neighbor)
env.new_random_game()  # initialize parameters in env


######################################################

"""
Environment로 부터 state를 얻는다.
"""
def get_state(env, idx=(0,0), ind_episode=1., epsi=0.02):
    """ Get state from the environment """

    # V2I_channel = (env.V2I_channels_with_fastfading[idx[0], :] - 80) / 60
    V2I_fast = (env.V2I_channels_with_fastfading[idx[0], :] - env.V2I_channels_abs[idx[0]] + 10)/35

    # V2V_channel = (env.V2V_channels_with_fastfading[:, env.vehicles[idx[0]].destinations[idx[1]], :] - 80) / 60
    V2V_fast = (env.V2V_channels_with_fastfading[:, env.vehicles[idx[0]].destinations[idx[1]], :] - env.V2V_channels_abs[:, env.vehicles[idx[0]].destinations[idx[1]]] + 10)/35

    V2V_interference = (-env.V2V_Interference_all[idx[0], idx[1], :] - 60) / 60

    V2I_abs = (env.V2I_channels_abs[idx[0]] - 80) / 60.0
    V2V_abs = (env.V2V_channels_abs[:, env.vehicles[idx[0]].destinations[idx[1]]] - 80)/60.0

    load_remaining = np.asarray([env.demand[idx[0], idx[1]] / env.demand_size])
    time_remaining = np.asarray([env.individual_time_limit[idx[0], idx[1]] / env.time_slow])

    # return np.concatenate((np.reshape(V2V_channel, -1), V2V_interference, V2I_abs, V2V_abs, time_remaining, load_remaining, np.asarray([ind_episode, epsi])))
    return np.concatenate((V2I_fast, np.reshape(V2V_fast, -1), V2V_interference, np.asarray([V2I_abs]), V2V_abs, time_remaining, load_remaining, np.asarray([ind_episode, epsi])))


# -----------------------------------------------------------

"""
DDPG로 구성된 모든 Agent는 Network를 500 250 120으로 구성한다.
output은 2개로 할 것임.
"""

n_input = len(get_state(env=env))
n_output = 2 #RB index, -100~23dBm
nVeh = 4
n_neighbor = 1
n_RB = n_veh


"""
지정한 agent가 행동함.
"""
def predict(agent, s_t, ep, test_ep = False, decay_epsilon = True):

    n_power_levels = len(env.V2V_power_dB_List)
    #print('epsilon-greedy : ', ep)
    if np.random.rand() < ep and not test_ep:
        #print('random select')
        pred_action = agent.random_action()
    else:
        #print('policy select')
        pred_action = agent.select_action(s_t, decay_epsilon=decay_epsilon)
        
    return pred_action

def predict_SAC(agent, s_t, test_ep = False, decay_epsilon = True, ):

    n_power_levels = len(env.V2V_power_dB_List)
    
    pred_action = agent.choose_action(s_t)
    
    pred_action[0] = np.clip(pred_action[0], 0, 3.9)
    
    pred_action[1] = np.clip(pred_action[1], -100.0, 23.0)
       
    return pred_action


"""
index 별 agent 저장 구현 완료
"""
def save_models(agent, model_path, agentindex, performanceInfo):
    """ Save models to the current directory with the name filename """
  
    saveFileName = 'agent' + str(agentindex) + '_' + performanceInfo
    agent.save_models(model_path + saveFileName)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch on TORCS with Multi-modal')

    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
    parser.add_argument('--hidden1', default=512, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2', default=256, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--prate', default=0.0001, type=float, help='policy net learning rate (only for DDPG)')
        
    parser.add_argument('--discount', default=0.99, type=float, help='')
    parser.add_argument('--bsize', default=256, type=int, help='minibatch size') # 
    parser.add_argument('--rmsize', default=100000, type=int, help='memory size')
    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument('--tau', default=0.003, type=float, help='moving average for target network')
    parser.add_argument('--ou_theta', default=0.15, type=float, help='noise theta')
    parser.add_argument('--ou_sigma', default=0.2, type=float, help='noise sigma') 
    parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu') 
    parser.add_argument('--target_update_step', default=20, type=int, help='target update step') 
    parser.add_argument('--useHardupdate', default=0, type=int, help='0 : use hard update, 1 : use soft update')     
    parser.add_argument('--validate_episodes', default=20, type=int, help='how many episode to perform during validate experiment')
    parser.add_argument('--max_episode_length', default=500, type=int, help='')

    parser.add_argument('--debug', dest='debug', action='store_true')

    parser.add_argument('--train_iter', default=6000, type=int, help='train iters each timestep') # 200000
    parser.add_argument('--warmup', default=2000, type=int, help='train iters each timestep') # 200000
    parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--seed', default=-1, type=int, help='')
    parser.add_argument('--resume', default='default', type=str, help='Resuming model path for testing')
       
    args = parser.parse_args()
    
    lr_actor = 0.005
    lr_critic = 0.001
    gamma = args.discount
    
    batch_size = args.bsize
    memory_size = args.rmsize
    
    tau = 0.001 # 소프트 업데이트의 타우
    
    sampling_only_until = 200


    nb_states = len(get_state(env=env))
    nb_actions = 2
    
    #DDPG 학습은 시간이 매우 오래 걸림
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(DEVICE)
    n_episode = args.train_iter
    warmup = args.warmup
    
    n_episode = 50
    warmup = 0
    
    n_step_per_episode = int(env.time_slow/env.time_fast)
    epsi_final = 0.02
    epsi_anneal_length = int(0.8*n_episode)
    mini_batch_step = n_step_per_episode
    target_update_step = n_step_per_episode*4
    
    isHardUpdate = False
    
    if args.useHardupdate == 1:
        isHardUpdate = True
    
    
    print('train iter -> ', n_episode)
           
    # --------------------------------------------------------------
    agents = []
   
    for ind_agent in range(n_veh * n_neighbor):  # initialize agents
        print("Initializing agent", ind_agent)
              
        agent = Agent(input_dims=nb_states, env=env,
            n_actions=nb_actions)
              
        agents.append(agent)        

    # ------------------------- Training -----------------------------
    record_reward = np.zeros([n_episode*n_step_per_episode, 1])
    record_value_loss = []
    record_policy_loss = []
    record_loss = []
    
    all_select_rb = np.zeros([n_episode*n_step_per_episode, 1])
    all_select_power = np.zeros([n_episode*n_step_per_episode, 1])

    selectStepCount = 0
    if IS_TRAIN:
        for i_episode in range(n_episode):
            print("-------------------------")
            print('Episode:', i_episode)
            
            if i_episode < epsi_anneal_length:
                epsi = 1 - i_episode * (1 - epsi_final) / (epsi_anneal_length - 1)  # epsilon decreases over each episode
            else:
                epsi = epsi_final
                
            if i_episode%100 == 0:
                env.renew_positions() # update vehicle position
                env.renew_neighbor()
                env.renew_channel() # update channel slow fading
                env.renew_channels_fastfading() # update channel fast fading
                     
                        
            env.demand = env.demand_size * np.ones((env.n_Veh, env.n_neighbor))
            env.individual_time_limit = env.time_slow * np.ones((env.n_Veh, env.n_neighbor))
            env.active_links = np.ones((env.n_Veh, env.n_neighbor), dtype='bool')
            
            
            
            
            for i_step in range(n_step_per_episode):
                selectStepCount += 1
                time_step = i_episode*n_step_per_episode + i_step
                state_old_all = []
                action_all = []
                action_all_training = np.zeros([n_veh, n_neighbor, 2], dtype='float')
                
                for i in range(n_veh):
                    for j in range(n_neighbor):
                        agentIndex = i*n_neighbor+j
                        state = get_state(env, [i, j], i_episode/(n_episode-1), epsi)      
                        state_old_all.append(state)
                       #state = to_tensor(state, size=(1,33)).to(DEVICE)
                        
                        #랜덤 선택 횟수를 만족했으면
                        if warmup <= selectStepCount:
                            action = predict_SAC(agents[agentIndex], state, epsi) 
                        else:
                            mu, sigma = 10.0, 25.0                           
                            power = np.random.normal(mu, sigma, 1)
                            power = np.clip(power, -100, 23.0)
                            rb = random.uniform(0, 3.99)
                            action= np.array([rb, power[0]])
                            
                        action_all.append(action)
                        action_all_training[i, j, 0] = action[0]  # chosen RB
                        action_all_training[i, j, 1] = action[1]  # power level
                        
                        all_select_rb[time_step] = action[0]
                        all_select_power[time_step] = action[1]
                    

                # All agents take actions simultaneously, obtain shared reward, and update the environment.
                
                action_temp = action_all_training.copy()
                train_reward = env.act_for_training(action_temp)
                record_reward[time_step] = train_reward

                env.renew_channels_fastfading()
                env.Compute_Interference(action_temp)
                
                """
                모든 agent가 state, action, reward, next_state 저장
                """
                                                            
                for i in range(n_veh):
                    for j in range(n_neighbor):
                        
                        agentIndex = i*n_neighbor+j
                        
                        state_old = state_old_all[agentIndex]
                        action = action_all[agentIndex]
                        train_reward_temp = train_reward
                        state_new = get_state(env, [i, j], i_episode/(n_episode-1), epsi)
                                                                    
                        agents[agentIndex].remember(torch.tensor(state_old).view(1,33), torch.tensor(action).view(1,2), 
                                       torch.tensor(train_reward_temp).view(1,1), torch.tensor(state_new).view(1,33), torch.tensor(1).view(1,1))
                        
                        # training this agent
                        if agents[agentIndex].memory.mem_cntr > batch_size:
                            #train agent
                            agents[agentIndex].learn()
                            
                            
                                
    
        print('Training Done. Saving models...')
        
        """
        agent 개별 저장
        """
        print('model save 시작')
        
        
        totalModelPath = './marl_model'
        totalModelPath_agent0 = './marl_model/agent_0/'
        totalModelPath_agent1 = './marl_model/agent_1/'
        totalModelPath_agent2 = './marl_model/agent_2/'
        totalModelPath_agent3 = './marl_model/agent_3/'
        
        createFolder(totalModelPath)
        createFolder(totalModelPath_agent0)
        createFolder(totalModelPath_agent1)
        createFolder(totalModelPath_agent2)
        createFolder(totalModelPath_agent3)
        
        for i in range(n_veh):
            for j in range(n_neighbor):
                model_path = './marl_model/agent_' + str(i * n_neighbor + j) + '/'
                save_models(agents[i*n_neighbor+j], model_path, i*n_neighbor+j, str(i_episode))
                
        print('model save 완료')
        
        print('log save 시작')
        
        current_dir = './'
        
        totalModelPath = './marl_model'
        all_select_rb = np.asarray(all_select_rb).reshape((-1, n_veh*n_neighbor))      
        MakeCSVFile(totalModelPath, 'selectRB.csv', ['agent0_sel_RB','agent1_selRB','agent2_selRB','agent3_selRB'],all_select_rb)
    
        totalModelPath = './marl_model'
        all_select_power = np.asarray(all_select_power).reshape((-1, n_veh*n_neighbor))      
        MakeCSVFile(totalModelPath, 'selectPower.csv', ['agent0_sel_Power','agent1_sel_Power','agent2_sel_Power','agent3_sel_Power'],all_select_power)
    
        record_reward = np.asarray(record_reward).reshape((-1, n_veh*n_neighbor))    
        reward_path = current_dir + 'marl_model/reward.mat'      
        MakeCSVFile(totalModelPath, 'reward.csv', ['reward0','reward1','reward2','reward3'],record_reward)
          
        record_value_loss = np.asarray(record_value_loss).reshape((-1, n_veh*n_neighbor))    
        loss_path = current_dir + 'marl_model/train_value_loss.mat'   
        MakeCSVFile(totalModelPath, 'train_value_loss.csv', ['loss0','loss1','loss2','loss3'],record_value_loss)
        print('log save 완료')
