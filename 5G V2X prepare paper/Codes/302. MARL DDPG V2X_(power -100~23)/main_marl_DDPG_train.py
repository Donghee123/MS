from __future__ import division, print_function
import random
import scipy
import scipy.io
import numpy as np
import Environment_marl
import os
import sys
import argparse
from ddpg import DDPG
from ddpgagent import Agent
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

def predict_using_warmup(agent, s_t, warmup_step, test_ep = False, decay_epsilon = True):

    n_power_levels = len(env.V2V_power_dB_List)
    
    if agent.select_step < warmup_step and not test_ep:
        #print('random select')
        pred_action = agent.random_action()
    else:
        #print('policy select')
        pred_action = agent.select_action(s_t, decay_epsilon=decay_epsilon)
        
    return pred_action



"""
DDPG update
"""
def DDPG_agent_learning(current_agent, isHardUpdate = False):   
    value_loss, policy_loss = current_agent.update_policy(isHardUpdate)
    return value_loss, policy_loss


"""
index 별 agent 저장 구현 완료
"""
def save_models(agent, model_path, agentindex, performanceInfo):
    """ Save models to the current directory with the name filename """
  
    saveFileName = str(agentindex) + '_' + performanceInfo
    agent.save_model(model_path, saveFileName)
    

"""
load agent 봉인!
"""
"""
def load_models(sess, model_path):
    Restore models from the current directory with the name filename

    dir_ = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(dir_, "model/" + model_path)
    saver.restore(sess, model_path)
"""




                            

    
"""
# -------------- Testing --------------
if IS_TEST:
    print("\nRestoring the model...")

    for i in range(n_veh):
        for j in range(n_neighbor):
            model_path = label + '/agent_' + str(i * n_neighbor + j)
            load_models(sesses[i * n_neighbor + j], model_path)

    V2I_rate_list = []
    V2V_success_list = []
    V2I_rate_list_rand = []
    V2V_success_list_rand = []
    rate_marl = np.zeros([n_episode_test, n_step_per_episode, n_veh, n_neighbor])
    rate_rand = np.zeros([n_episode_test, n_step_per_episode, n_veh, n_neighbor])
    demand_marl = env.demand_size * np.ones([n_episode_test, n_step_per_episode+1, n_veh, n_neighbor])
    demand_rand = env.demand_size * np.ones([n_episode_test, n_step_per_episode+1, n_veh, n_neighbor])
    power_rand = np.zeros([n_episode_test, n_step_per_episode, n_veh, n_neighbor])
    for idx_episode in range(n_episode_test):
        print('----- Episode', idx_episode, '-----')

        env.renew_positions()
        env.renew_neighbor()
        env.renew_channel()
        env.renew_channels_fastfading()

        env.demand = env.demand_size * np.ones((env.n_Veh, env.n_neighbor))
        env.individual_time_limit = env.time_slow * np.ones((env.n_Veh, env.n_neighbor))
        env.active_links = np.ones((env.n_Veh, env.n_neighbor), dtype='bool')

        env.demand_rand = env.demand_size * np.ones((env.n_Veh, env.n_neighbor))
        env.individual_time_limit_rand = env.time_slow * np.ones((env.n_Veh, env.n_neighbor))
        env.active_links_rand = np.ones((env.n_Veh, env.n_neighbor), dtype='bool')

        V2I_rate_per_episode = []
        V2I_rate_per_episode_rand = []
        for test_step in range(n_step_per_episode):
            # trained models
            action_all_testing = np.zeros([n_veh, n_neighbor, 2], dtype='int32')
            for i in range(n_veh):
                for j in range(n_neighbor):
                    state_old = get_state(env, [i, j], 1, epsi_final)
                    action = predict(sesses[i*n_neighbor+j], state_old, epsi_final, True)
                    action_all_testing[i, j, 0] = action % n_RB  # chosen RB
                    action_all_testing[i, j, 1] = int(np.floor(action / n_RB))  # power level

            action_temp = action_all_testing.copy()
            V2I_rate, V2V_success, V2V_rate = env.act_for_testing(action_temp)
            V2I_rate_per_episode.append(np.sum(V2I_rate))  # sum V2I rate in bps
            rate_marl[idx_episode, test_step,:,:] = V2V_rate
            demand_marl[idx_episode, test_step+1,:,:] = env.demand

            # random baseline
            action_rand = np.zeros([n_veh, n_neighbor, 2], dtype='int32')
            action_rand[:, :, 0] = np.random.randint(0, n_RB, [n_veh, n_neighbor]) # band
            action_rand[:, :, 1] = np.random.randint(0, len(env.V2V_power_dB_List), [n_veh, n_neighbor]) # power

            V2I_rate_rand, V2V_success_rand, V2V_rate_rand = env.act_for_testing_rand(action_rand)
            V2I_rate_per_episode_rand.append(np.sum(V2I_rate_rand))  # sum V2I rate in bps
            rate_rand[idx_episode, test_step, :, :] = V2V_rate_rand
            demand_rand[idx_episode, test_step+1,:,:] = env.demand_rand
            for i in range(n_veh):
                for j in range(n_neighbor):
                    power_rand[idx_episode, test_step, i, j] = env.V2V_power_dB_List[int(action_rand[i, j, 1])]

            # update the environment and compute interference
            env.renew_channels_fastfading()
            env.Compute_Interference(action_temp)

            if test_step == n_step_per_episode - 1:
                V2V_success_list.append(V2V_success)
                V2V_success_list_rand.append(V2V_success_rand)

        V2I_rate_list.append(np.mean(V2I_rate_per_episode))
        V2I_rate_list_rand.append(np.mean(V2I_rate_per_episode_rand))

        print(round(np.average(V2I_rate_per_episode), 2), 'rand', round(np.average(V2I_rate_per_episode_rand), 2))
        print(V2V_success_list[idx_episode], 'rand', V2V_success_list_rand[idx_episode])

    print('-------- marl -------------')
    print('n_veh:', n_veh, ', n_neighbor:', n_neighbor)
    print('Sum V2I rate:', round(np.average(V2I_rate_list), 2), 'Mbps')
    print('Pr(V2V success):', round(np.average(V2V_success_list), 4))

    print('-------- random -------------')
    print('n_veh:', n_veh, ', n_neighbor:', n_neighbor)
    print('Sum V2I rate:', round(np.average(V2I_rate_list_rand), 2), 'Mbps')
    print('Pr(V2V success):', round(np.average(V2V_success_list_rand), 4))

    with open("Data.txt", "a") as f:
        f.write('-------- marl, ' + label + '------\n')
        f.write('n_veh: ' + str(n_veh) + ', n_neighbor: ' + str(n_neighbor) + '\n')
        f.write('Sum V2I rate: ' + str(round(np.average(V2I_rate_list), 5)) + ' Mbps\n')
        f.write('Pr(V2V): ' + str(round(np.average(V2V_success_list), 5)) + '\n')
        f.write('--------random ------------\n')
        f.write('Rand Sum V2I rate: ' + str(round(np.average(V2I_rate_list_rand), 5)) + ' Mbps\n')
        f.write('Rand Pr(V2V): ' + str(round(np.average(V2V_success_list_rand), 5)) + '\n')

    current_dir = os.path.dirname(os.path.realpath(__file__))
    marl_path = os.path.join(current_dir, "model/" + label + '/rate_marl.mat')
    scipy.io.savemat(marl_path, {'rate_marl': rate_marl})
    rand_path = os.path.join(current_dir, "model/" + label + '/rate_rand.mat')
    scipy.io.savemat(rand_path, {'rate_rand': rate_rand})

    demand_marl_path = os.path.join(current_dir, "model/" + label + '/demand_marl.mat')
    scipy.io.savemat(demand_marl_path, {'demand_marl': demand_marl})
    demand_rand_path = os.path.join(current_dir, "model/" + label + '/demand_rand.mat')
    scipy.io.savemat(demand_rand_path, {'demand_rand': demand_rand})

    power_rand_path = os.path.join(current_dir, "model/" + label + '/power_rand.mat')
    scipy.io.savemat(power_rand_path, {'power_rand': power_rand})


# close sessions
for sess in sesses:
    sess.close()
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch on TORCS with Multi-modal')

    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
    parser.add_argument('--hidden1', default=512, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2', default=256, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--prate', default=0.0001, type=float, help='policy net learning rate (only for DDPG)')
        
    parser.add_argument('--discount', default=0.9, type=float, help='')
    parser.add_argument('--bsize', default=512, type=int, help='minibatch size') # 
    parser.add_argument('--rmsize', default=6000000, type=int, help='memory size')
    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument('--tau', default=0.003, type=float, help='moving average for target network')
    parser.add_argument('--ou_theta', default=0.15, type=float, help='noise theta')
    parser.add_argument('--ou_sigma', default=0.2, type=float, help='noise sigma') 
    parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu') 
    parser.add_argument('--target_update_step', default=20, type=int, help='target update step') 
    parser.add_argument('--useHardupdate', default=0, type=int, help='0 : use hard update, 1 : use soft update')     
    parser.add_argument('--validate_episodes', default=20, type=int, help='how many episode to perform during validate experiment')
    parser.add_argument('--max_episode_length', default=500, type=int, help='')

    parser.add_argument('--output', default='output', type=str, help='')
    parser.add_argument('--debug', dest='debug', action='store_true')
        
    parser.add_argument('--init_w', default=0.001, type=float, help='') 
    parser.add_argument('--warmup', default=10000, type=int, help='time without training but only filling the replay memory') # 10000
    parser.add_argument('--validate_steps', default=2000, type=int, help='how many steps to perform a validate experiment') # 2000
    parser.add_argument('--train_iter', default=3000, type=int, help='train iters each timestep') # 200000
    parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--seed', default=-1, type=int, help='')
    parser.add_argument('--resume', default='default', type=str, help='Resuming model path for testing')
        
    nb_states = len(get_state(env=env))
    nb_actions = 2

    args = parser.parse_args()

    warmup_step = args.warmup
    n_episode = args.train_iter
    n_step_per_episode = int(env.time_slow/env.time_fast)
    epsi_final = 0.02
    epsi_anneal_length = int(0.8*n_episode)
    mini_batch_step = n_step_per_episode
    target_update_step = n_step_per_episode*4
    
    isHardUpdate = False
    
    if args.useHardupdate == 1:
        isHardUpdate = True
    
    print('use update mode 0 : soft, 1: hard -> ', args.useHardupdate)
    print('use warmupstep -> ', warmup_step)
    print('train iter -> ', n_episode)
           
    # --------------------------------------------------------------
    agents = []

    for ind_agent in range(n_veh * n_neighbor):  # initialize agents
        print("Initializing agent", ind_agent)
        DDPGModel = DDPG(nb_states, nb_actions, args, n_veh)
        agent = Agent(DDPGModel)
        agents.append(agent)

    # ------------------------- Training -----------------------------
    record_reward = np.zeros([n_episode*n_step_per_episode, 1])
    record_value_loss = []
    record_policy_loss = []
    record_loss = []

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
                
                for i in range(n_veh):
                    for j in range(n_neighbor):                       
                        agents[i*n_neighbor+j].reset()
                        
                        
            env.demand = env.demand_size * np.ones((env.n_Veh, env.n_neighbor))
            env.individual_time_limit = env.time_slow * np.ones((env.n_Veh, env.n_neighbor))
            env.active_links = np.ones((env.n_Veh, env.n_neighbor), dtype='bool')

            for i_step in range(n_step_per_episode):
                time_step = i_episode*n_step_per_episode + i_step
                state_old_all = []
                action_all = []
                action_all_training = np.zeros([n_veh, n_neighbor, 2], dtype='float')
                
                for i in range(n_veh):
                    for j in range(n_neighbor):
                        state = get_state(env, [i, j], i_episode/(n_episode-1), epsi)      
                        state_old_all.append(state)
                        action = predict_using_warmup(agents[i*n_neighbor+j], state, warmup_step, epsi) 
                        action_all.append(action)
                        action_all_training[i, j, 0] = action[0]  # chosen RB
                        action_all_training[i, j, 1] = action[1]  # power level

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
                        state_old = state_old_all[n_neighbor * i + j]
                        action = action_all[n_neighbor * i + j]
                        state_new = get_state(env, [i, j], i_episode/(n_episode-1), epsi)
                        
                        agents[i*n_neighbor+j].reset_state(state_old) #state 저장                   
                        agents[i*n_neighbor+j].observe(train_reward, state_new) #action, reward, next state 모두 저장.
                        
                        # training this agent
                        if time_step % mini_batch_step == mini_batch_step-1:
                            loss_val_batch, loss_policy_batch = DDPG_agent_learning(agents[i*n_neighbor+j], isHardUpdate=isHardUpdate)
                            
                            record_value_loss.append(loss_val_batch)
                            record_policy_loss.append(loss_policy_batch)
                            
                            if i == 0 and j == 0:
                                print('step:', time_step, 'agent',i*n_neighbor+j, 'loss', loss_val_batch)
                                
    
        print('Training Done. Saving models...')
        
        """
        agent 개별 저장
        """
        print('model save 시작')
        
        
        totalModelPath = './marl_model'
        totalModelPath_agent0 = './marl_model/agent_0'
        totalModelPath_agent1 = './marl_model/agent_1'
        totalModelPath_agent2 = './marl_model/agent_2'
        totalModelPath_agent3 = './marl_model/agent_3'
        
        createFolder(totalModelPath)
        createFolder(totalModelPath_agent0)
        createFolder(totalModelPath_agent1)
        createFolder(totalModelPath_agent2)
        createFolder(totalModelPath_agent3)
        
        for i in range(n_veh):
            for j in range(n_neighbor):
                model_path = './marl_model/agent_' + str(i * n_neighbor + j)
                save_models(agents[i*n_neighbor+j], model_path, i*n_neighbor+j, str(i_episode))
                
        print('model save 완료')
        
        print('log save 시작')
        
        current_dir = './'
        
        record_reward = np.asarray(record_reward).reshape((-1, n_veh*n_neighbor))    
        reward_path = current_dir + 'marl_model/reward.mat'    
        scipy.io.savemat(reward_path, {'reward': record_reward})   
        MakeCSVFile(totalModelPath, 'reward.csv', ['reward0','reward1','reward2','reward3'],record_reward)
          
        record_value_loss = np.asarray(record_value_loss).reshape((-1, n_veh*n_neighbor))    
        loss_path = current_dir + 'marl_model/train_value_loss.mat'   
        scipy.io.savemat(loss_path, {'train_value_loss': record_value_loss})
        MakeCSVFile(totalModelPath, 'train_value_loss.csv', ['loss0','loss1','loss2','loss3'],record_value_loss)
        print('log save 완료')
