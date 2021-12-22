#!/usr/bin/env python3 
import numpy as np
import argparse
from copy import deepcopy
import torch
import gym
import matplotlib.pyplot as plt

from MLP import MultiLayerPerceptron as MLP
from DQN import prepare_training_inputs
from DDPG import DDPG, Actor, Critic
from DDPG import OrnsteinUhlenbeckProcess as OUProcess

from common.target_update import soft_update
from common.train_utils import to_tensor
from common.memory.memory import ReplayMemory

from Environment import *
from util import *
import pandas as pd
import csv

#File 유틸 함수들    
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

def showSelectHistGraph(listOfRB, lustOfPower):
    plt.subplot(211)
    plt.hist(listOfRB, histtype='step')
    plt.title('Left : RB Rate, Right : Power Rate')
    plt.subplot(212)
    plt.hist(lustOfPower, histtype='step')
    plt.show()
        
def MakeCSVFile(strFolderPath, strFilePath, aryOfHedaers, aryOfDatas):
    strTotalPath = "%s\%s" % (strFolderPath,strFilePath)
    
    f = open(strTotalPath,'w', newline='')
    wr = csv.writer(f)
    wr.writerow(aryOfHedaers)
    
    for i in range(0,len(aryOfDatas)):
        wr.writerow(aryOfDatas[i])
    
    f.close()

#gym.undo_logger_setup()
def GetRB_Power(powerMin,action):
    
    powerRange = 23.0 - powerMin
    
    selectedRBIndex = np.argmax(action[0:20])
    actionFromPolicy = ((action[20] + 2.0) * (powerRange / 4)) + powerMin
    actionFromPolicy = np.clip(actionFromPolicy, powerMin, 23.0)
    selectedPower = actionFromPolicy
    
    return selectedRBIndex, selectedPower

def train_V2(args, memory, agent, env):
    
    isShowTestGraph = args.showtestGraph
    
    minPower = args.power_min
    totalEpisode = args.train_iter
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
          
    selectStep = 0  
    episode = 0
    
    V2I_Rate_list = np.zeros(totalEpisode)
    V2V_Rate_list = np.zeros(totalEpisode)
    Fail_percent_list = np.zeros(totalEpisode)
    
    while episode < totalEpisode:
        print('===========================')
        print('curEpisode : ', episode)
        
                    
        #for smple data 
        listOfselRB = []
        listOfselPower = []
        listOfReward = []
        listofState_nextState = []
        
        env.new_random_game(env.n_Veh)
        
        #차량 position 초기화 반복문
        env.preFixedUpdateCount = 0
        env.renew_positions_using_fixed_data()
        
        # OU random process 리셋     
        ou_noise = OUProcess(mu=np.zeros(1))          
        cum_r = 0
                                                                   
        V2IRate_list = []
        V2VRate_list = []                    
        selPowerRateList = []
        selRBRateList = []
        percent = 0 
        
        #agent 값들 모두 초기
        agent.action_all_with_power[:,:,0] = 0
        agent.action_all_with_power[:,:,1] = 0
        
        #episode 테스트 시작.
        test_sample = 200
        
        for k in range(test_sample):
            action_temp = agent.action_all_with_power.copy()
                      
            for i in range(len(env.vehicles)):
                selectStep += 1
                
                agent.action_all_with_power[i,:,0] = -1
                sorted_idx = np.argsort(env.individual_time_limit[i,:])      
                
                for j in sorted_idx:     
                    
                    state = env.get_state(idx = [i,j], isTraining = False, action_all_with_power_training = None, action_all_with_power = agent.action_all_with_power)                           
                    state = to_tensor(state)
                                      
                    action = agent.get_action(state).cpu().numpy() + ou_noise()[0]
                    selRBIndex, selPowerdBm = GetRB_Power(minPower, action)
                    print('RB, Power' , [selRBIndex, selPowerdBm])
                    selRBRateList.append(selRBIndex)
                    selPowerRateList.append(selPowerdBm)
                    
                    agent.action_all_with_power[i, j, 0] = selRBIndex
                    agent.action_all_with_power[i, j, 1] = selPowerdBm

                    action_temp = agent.action_all_with_power.copy()
                    V2IRate, V2VRate, percent = env.act_asyn_train(action_temp) #self.action_all)            
                    V2IRate_list.append(np.sum(V2IRate))
                    V2VRate_list.append(np.sum(V2VRate))
                    nest_state = env.get_state(idx = [i,j], isTraining = False, action_all_with_power_training = None, action_all_with_power = agent.action_all_with_power) 
                    reward = -percent
                    done = False
                    
                    experience = (state.view(1,82),
                                  torch.tensor(action).view(1,21),
                                  torch.tensor(reward).view(1,1),
                                  torch.tensor(nest_state).view(1,82),
                                  torch.tensor(done).view(1,1))
                    
                    memory.push(experience)
                    cum_r += reward
                    
                    if len(memory) >= sampling_only_until:
                        #train agent
                        sampled_exps = memory.sample(batch_size)
                        sampled_exps = prepare_training_inputs(sampled_exps, device = DEVICE)
                        agent.update(*sampled_exps)
                      
                        #update target networks
                        #OUProcess에 의해 noisy 한 환경이 생김.
                        #soft_update를 통해 noisy 한 환경에서 tau값을 조절하며 업데이트함
                        soft_update(agent.actor, agent.actor_target, tau)
                        soft_update(agent.critic, agent.critic_target, tau)  
                                                            
        listOfReward.append(cum_r)            
                                               
        if (isShowTestGraph == 1):
           showSelectHistGraph(selRBRateList, selPowerRateList)
                
        V2I_Rate_list[episode] = np.mean(np.asarray(V2IRate_list))
        V2V_Rate_list[episode] = np.mean(np.asarray(V2VRate_list))
        Fail_percent_list[episode] = percent
        
        episode += 1
        
        print('cumulative reward :', cum_r)
        print('failure probability is, ', percent)
                    
        #episode 마 할때마다 모델 저장
        savePath = 'ddpg/model/'
        performanceInfo = str(episode) + '_' + str(np.mean(V2IRate_list)) + '_' + str(np.mean(V2VRate_list)) + '_' + str(percent)        
        criticPath = savePath + 'critic_' + performanceInfo
        actorPath = savePath + 'actor_' + performanceInfo
        
        torch.save(agent.critic_target.state_dict(), criticPath)       
        torch.save(agent.actor_target.state_dict(), actorPath)    
        
        #agent.save_model('ddpg/model', performanceInfo)
        print ('The number of vehicle is ', len(env.vehicles))
        print ('Mean of the V2I rate + V2V rate is that ', np.mean(V2IRate_list) + np.mean(V2VRate_list))
        print ('Mean of the V2I rate is that ', np.mean(V2IRate_list))
        print ('Mean of the V2V rate is that ', np.mean(V2VRate_list))
        print('Episode End Fail percent is that ', percent) 



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch on TORCS with Multi-modal')

    parser.add_argument('--mode', default='train_V2', type=str, help='support option: train/test')
    parser.add_argument('--train_resume', default=0, type=int, help='train resume, using path : ./ddpg/resume model/actor, ./ddpg/resume model/critic')
    parser.add_argument('--using_position_data', default=1, type=int, help='using test data, using path : ./position/vehiclePosition.csv')
    parser.add_argument('--hidden1', default=256, type=int, help='hidden1 num of first fully connect layer')
    parser.add_argument('--hidden2', default=128, type=int, help='hidden2 num of second fully connect layer')
    parser.add_argument('--hidden3', default=64, type=int, help='hidden3 num of first fully connect layer')

    parser.add_argument('--lr_actor', default=0.005, type=float, help='Actor learning rate')
    parser.add_argument('--lr_critic', default=0.001, type=float, help='Critic learning rate')
    parser.add_argument('--gamma', default=0.99, type=float, help='')   
    parser.add_argument('--bsize', default=256, type=int, help='minibatch size') #256
    parser.add_argument('--rmsize', default=50000, type=int, help='memory size')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--sampling_only_until', default=260, type=int, help='train start step') # 2000

    
    parser.add_argument('--power_min', default=-10.0, type=float, help='') 
       
    parser.add_argument('--train_iter', default=100, type=int, help='train iters each timestep') # 4000
    parser.add_argument('--seed', default=-1, type=int, help='')
    parser.add_argument('--showtestGraph', default='0', type=int, help='Show graph for test')

    args = parser.parse_args()
     
    up_lanes = [3.5/2, 3.5/2 + 3.5, 250+3.5/2,
            250+3.5+3.5/2, 500+3.5/2, 500+3.5+3.5/2]
    down_lanes = [250-3.5-3.5/2, 250-3.5/2, 500-3.5 -
              3.5/2, 500-3.5/2, 750-3.5-3.5/2, 750-3.5/2]
    left_lanes = [3.5/2, 3.5/2 + 3.5, 433+3.5/2,
              433+3.5+3.5/2, 866+3.5/2, 866+3.5+3.5/2]
    right_lanes = [433-3.5-3.5/2, 433-3.5/2, 866-3.5 -
               3.5/2, 866-3.5/2, 1299-3.5-3.5/2, 1299-3.5/2]

    width = 750
    height = 1299
    nVeh = 60

    # V2X 환경 적용
    env = Environ(down_lanes, up_lanes, left_lanes,
              right_lanes, width, height, nVeh, args.power_min)  # V2X 환경 생성
    
    if args.using_position_data == 1:
        env.load_position_data('./position/vehiclePosition.csv')
        
    if args.seed > 0:
        np.random.seed(args.seed)
        env.seed(args.seed)

    nb_states = env.observation_space.shape[0]
    nb_actions = env.action_space.shape[0]
    
    lr_actor = args.lr_actor
    lr_critic = args.lr_critic
    gamma = args.gamma
    batch_size = args.bsize
    memory_size = args.rmsize
    tau = args.tau
    sampling_only_until = args.sampling_only_until
    
    hidden1 = args.hidden1
    hidden2 = args.hidden2
    hidden3 = args.hidden3
    
    #init network
    actor = Actor(nb_states, nb_actions, num_neurons=[hidden1,hidden2,hidden3])
    actor_target = Actor(nb_states, nb_actions, num_neurons=[hidden1,hidden2,hidden3])
    critic = Critic(nb_states, nb_actions, num_neurons=[hidden1,hidden2,hidden3])
    critic_target = Critic(nb_states, nb_actions, num_neurons=[hidden1,hidden2,hidden3])

    agent = DDPG(nVeh, critic=critic,critic_target=critic_target,actor=actor, actor_target=actor_target,
                 lr_actor=lr_actor, lr_critic=lr_critic, gamma = gamma)
    
    
    memory = ReplayMemory(memory_size)
    
    if args.train_resume == 1:
        actor.mlp.load_state_dict(torch.load('./ddpg/resume model/actor'))
        actor_target.mlp.load_state_dict(torch.load('./ddpg/resume model/actor'))
        critic.mlp.load_state_dict(torch.load('./ddpg/resume model/critic'))
        critic_target.mlp.load_state_dict(torch.load('./ddpg/resume model/critic'))
        
   
    train_V2(args, memory, agent, env)
     

    
