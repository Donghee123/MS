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
from sac_torch import Agent as Agent_sac
import pandas as pd
import csv

from MLP import MultiLayerPerceptron as MLP
from common.train_utils import to_tensor
from common.memory.memory import ReplayMemory
from DQN import DQN

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
    
def prepare_training_inputs(sampled_exps, device='cpu'):
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []
    
    for sampled_exp in sampled_exps:
        states.append(sampled_exp[0])
        actions.append(sampled_exp[1])
        rewards.append(sampled_exp[2])
        next_states.append(sampled_exp[3])
        dones.append(sampled_exp[4])

    states = torch.Tensor(states).float().to(device)
    #states = torch.cat(states, dim=0).float().to(device)
    actions = torch.cat(actions, dim=0).to(device)
    rewards = torch.cat(rewards, dim=0).float().to(device)
    next_states = torch.Tensor(next_states).float().to(device)
    #next_states = torch.cat(next_states, dim=1).float().to(device)
    dones = torch.cat(dones, dim=0).float().to(device)
    return states, actions, rewards, next_states, dones

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


nVeh = 4
n_neighbor = 1
n_RB = n_veh


"""
지정한 agent가 행동함.
"""
def predict_dqn(agent, s_t, ep, test_ep = False):

    if np.random.rand() < ep and not test_ep:
        pred_action = np.random.randint(4)
    else:
        pred_action = agent.get_action(s_t)
        
    return pred_action

def predict_SAC(agent, s_t, test_ep = False, decay_epsilon = True, ):

    
    pred_action = agent.choose_action(s_t)
    
    pred_action[0] = np.clip(pred_action[0], -100.0, 23.0)
        
    return pred_action


"""
index 별 agent 저장 구현 완료
"""
def save_models_sac(agent, model_path, agentindex, performanceInfo):
    """ Save models to the current directory with the name filename """
  
    saveFileName = 'agent' + str(agentindex) + '_' + performanceInfo
    agent.save_models(model_path + saveFileName)
    
def save_models_dqn(agent, model_path, agentindex, performanceInfo):
    """ Save models to the current directory with the name filename """
    """
    current_dir = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(current_dir, "model/" + model_path)
    
    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))
    """
    
    saveFileName = 'agent' + str(agentindex) + '_' + performanceInfo
    #target, main network 저장
    torch.save(agent.qnet.state_dict(), model_path + saveFileName + '_main')
    torch.save(agent.qnet_target.state_dict(), model_path + saveFileName + '_target')

def load_models_dqn(agent, model_path):
    """ Restore models from the current directory with the name filename """

    dir_ = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(dir_, "model/" + model_path)
    
    #target, main network 읽기
    agent.qnet.load_state_dict(torch.load(model_path + '_main'))
    agent.qnet_target.load_state_dict(torch.load(model_path + '_target'))
    

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
    parser.add_argument('--lr', default=0.001, type=float, help='train iters each timestep') # 200000
    parser.add_argument('--warmup', default=2000, type=int, help='train iters each timestep') # 200000
    parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--seed', default=-1, type=int, help='')
    parser.add_argument('--resume', default='default', type=str, help='Resuming model path for testing')
       
    args = parser.parse_args()
    
   
    n_hidden_1 = 500
    n_hidden_2 = 250
    n_hidden_3 = 120   
    
    #DQN
    nb_states_dqn = len(get_state(env=env))
    nb_actions_dqn = 1
    sampling_only_util = 2000
    lr = args.lr

    total_eps = 3000
    eps_max = 0.08
    eps_min = 0.01 
    target_update_interval = 110
    #
    
    #SAC
    nb_states_sac = nb_states_dqn + 1
    nb_actions_sac = 1
    
    lr_actor = args.lr
    lr_critic = args.lr
    gamma = args.discount
    
    batch_size = args.bsize
    memory_size = args.rmsize
    
    tau = 0.001 # 소프트 업데이트의 타우
       
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(DEVICE)
    n_episode = args.train_iter
    warmup = args.warmup
    
    n_step_per_episode = int(env.time_slow/env.time_fast)
    epsi_final = 0.02
    epsi_anneal_length = int(0.8*n_episode)
    mini_batch_step = n_step_per_episode
    target_update_step = n_step_per_episode*4
    #
    
    isHardUpdate = False
    
    if args.useHardupdate == 1:
        isHardUpdate = True
    
    
    print('train iter -> ', n_episode)
           
    # --------------------------------------------------------------
    dqn_agents = []
    sac_agents = []
    dqn_memory = []
    for ind_agent in range(n_veh * n_neighbor):  # initialize agents
        print("Initializing agent", ind_agent)
        
        #for SAC
        agent = Agent_sac(input_dims=nb_states_sac, env=env,
            n_actions=nb_actions_sac)
        
        sac_agents.append(agent)  
        
        #for DQN
        qnet = MLP(nb_states_dqn,nb_actions_dqn, num_neurons=[n_hidden_1, n_hidden_2, n_hidden_3])
        qnet_target = MLP(nb_states_dqn, nb_actions_dqn, num_neurons=[n_hidden_1, n_hidden_2, n_hidden_3])
        qnet_target.load_state_dict(qnet.state_dict())
        dqn_agent = DQN(nb_states_dqn, nb_actions_dqn, qnet = qnet, qnet_target = qnet_target, lr=lr, gamma=gamma, epsilon=1.0)
        dqn_agents.append(dqn_agent)
        memory = ReplayMemory(memory_size)
        dqn_memory.append(memory)

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
                                                
                        #DQN action
                        state_dqn = to_tensor(state, size=(1, nb_states_dqn))
                        action_dqn = predict_dqn(dqn_agents[agentIndex], state_dqn, epsi)
                        
                        state_old_all.append(state)
                        #state = to_tensor(state, size=(1,33)).to(DEVICE)
                        
                                                                   
                        #SAC action                                 
                        state_sac = np.append(state, np.array([action_dqn]))                     
                        action_sac = predict_SAC(sac_agents[agentIndex], state_sac, epsi) 
                                       
                        action_all.append(np.array([action_dqn, action_sac]))
                        action_all_training[i, j, 0] = action_dqn  # chosen RB
                        action_all_training[i, j, 1] = action_sac  # power level
                        
                        all_select_rb[time_step] = action_dqn
                        all_select_power[time_step] = action_sac
                    

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
                        
                        #DQN 
                        state_old_dqn = state_old_all[agentIndex]
                        action_dqn = action_all[agentIndex][0]
                        state_new_dqn = get_state(env, [i, j], i_episode/(n_episode-1), epsi)
                        #state_new_dqn = to_tensor(state_new_dqn, size=(1, nb_states_dqn))
                        
                        #1개의 경험 튜플
                        experience = (state_old_dqn, 
                                      torch.tensor(action_dqn).view(1,1),
                                      torch.tensor(train_reward / 100.0).view(1,1),#r값을 스케일링함
                                      state_new_dqn,
                                      torch.tensor(False).view(1,1))
                        
                        #각 agent들의 memory에 저장
                        dqn_memory[agentIndex].push(experience)  # add entry to this agent's memory
                        
                        #SAC
                        state_old_sac = state_old_all[agentIndex]
                        state_old_sac = np.append(state_old_sac, np.array([action_all[agentIndex][0]]))
                        action_sac = action_all[agentIndex][1]
                        train_reward_temp = train_reward
                        state_new_dqn = get_state(env, [i, j], i_episode/(n_episode-1), epsi)
                        
                        state_new_dqn = to_tensor(state, size=(1, nb_states_dqn))
                        
                        next_action_dqn = predict_dqn(dqn_agents[agentIndex], state_new_dqn , epsi)
                        state_new_sac =  np.append(state_new_dqn, np.array([next_action_dqn]))  
                        
                        sac_agents[agentIndex].remember(torch.tensor(state_old_sac).view(1,34), torch.tensor(action_sac).view(1,1), 
                                       torch.tensor(train_reward_temp).view(1,1), torch.tensor(state_new_sac).view(1,34), torch.tensor(1).view(1,1))
                        
                        # training this agent
                        if sac_agents[agentIndex].memory.mem_cntr > batch_size:
                            #train agent
                            sac_agents[agentIndex].learn()
                            
                 
                #모든 dqn Agent update
                for i in range(n_veh):
                    for j in range(n_neighbor):
                        
                        #메모리의 값이 충분히 찼다면 업데이트를 시킴
                        if len(memory) >= sampling_only_util:
                            sampled_exps = dqn_memory[i*n_neighbor+j].sample(batch_size)
                            sampled_exps = prepare_training_inputs(sampled_exps)    
                            dqn_agents[i*n_neighbor+j].update(*sampled_exps)
                                
                        #qnet의 파라미터를 하이퍼 파라미터 수만큼 업데이트 했다면 target qnet에 qnet의 파라미터를 업데이트함
                        if i_step % target_update_interval == 0: 
                            dqn_agents[i*n_neighbor+j].qnet_target.load_state_dict( dqn_agents[i*n_neighbor+j].qnet.state_dict() )
                                
    
        print('Training Done. Saving models...')
        
        """
        agent 개별 저장
        """
        print('model save 시작')
        
        
        totalModelPath = './marl_model'
        totalModelPath_agent0 = './marl_model/agent_sac0/'
        totalModelPath_agent1 = './marl_model/agent_sac1/'
        totalModelPath_agent2 = './marl_model/agent_sac2/'
        totalModelPath_agent3 = './marl_model/agent_sac3/'
        createFolder(totalModelPath)
        createFolder(totalModelPath_agent0)
        createFolder(totalModelPath_agent1)
        createFolder(totalModelPath_agent2)
        createFolder(totalModelPath_agent3)
        
        totalModelPath_agent0 = './marl_model/agent_dqn0/'
        totalModelPath_agent1 = './marl_model/agent_dqn1/'
        totalModelPath_agent2 = './marl_model/agent_dqn2/'
        totalModelPath_agent3 = './marl_model/agent_dqn3/'
        
        createFolder(totalModelPath_agent0)
        createFolder(totalModelPath_agent1)
        createFolder(totalModelPath_agent2)
        createFolder(totalModelPath_agent3)
        
        for i in range(n_veh):
            for j in range(n_neighbor):
                model_path = './marl_model/agent_sac' + str(i * n_neighbor + j) + '/'
                save_models_sac(sac_agents[i*n_neighbor+j], model_path, i*n_neighbor+j, str(i_episode))
                model_path = './marl_model/agent_dqn' + str(i * n_neighbor + j) + '/'
                save_models_dqn(dqn_agents[i*n_neighbor+j], model_path, i*n_neighbor+j, str(i_episode))
                
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
