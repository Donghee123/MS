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
    
    select_maxValue  = np.max(action[0:20])
    candidateList_selRB = np.where(action[0:20] == select_maxValue)
    selectedRBIndex = random.sample(list(candidateList_selRB[0]),k=1)[0]
    
    actionFromPolicy = ((action[20] + 2.0) * (powerRange / 4)) + powerMin
    actionFromPolicy = np.clip(actionFromPolicy, powerMin, 23.0)
    selectedPower = actionFromPolicy
    
    return selectedRBIndex, selectedPower

def train(args, memory, agent, env, fPowermin):
    
    isShowTestGraph = args.showtestGraph
    env.new_random_game(20)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'         
    for step in (range(0, args.train_iter)):
        
        if step == 0:       
            ou_noise = OUProcess(mu=np.zeros(1))       
            cum_r = 0             
                
        # prediction
        # action = self.predict(self.history.get())
        if (step % 2000 == 1):
           ou_noise = OUProcess(mu=np.zeros(1))  
           print('reward per 2000 step : ', cum_r)
           cum_r = 0
           env.new_random_game(20)
        
        isTraining = True
        print(step)
        
        for k in range(1):
            #i번째 송신 차량
            for i in range(len(env.vehicles)):
                #i번째 송신 차량에서 전송한 신호를 수신하는 j번째 수신 차량 
                for j in range(3): 
                    # i번째 차량에서 j번째 차량으로 데이터를 전송할때 state를 가져옴.
                    state = env.get_state(idx = [i,j], isTraining = isTraining, action_all_with_power_training = agent.action_all_with_power_training, action_all_with_power = agent.action_all_with_power)   
                    state = to_tensor(state)    
                    #state를 보고 action을 정함
                    #action은 선택한 power level, 선택한 resource block 정보를 가짐 
                    action = agent.get_action(state).cpu().numpy() + ou_noise()[0]  
                                          
                    selRBIndex, selPowerdBm = GetRB_Power(fPowermin, action)    
                    #선택한 resource block을 넣어줌
                    agent.action_all_with_power_training[i, j, 0] = selRBIndex
                     
                    #선택한 power level을 넣어줌
                    agent.action_all_with_power_training[i, j, 1] = selPowerdBm 
                                         
                    #선택한 power level과 resource block을 기반으로 reward를 계산함.
                    reward = env.act_for_training(agent.action_all_with_power_training, [i,j]) 
                        
                    next_state = env.get_state(idx = [i,j], isTraining = isTraining, action_all_with_power_training = agent.action_all_with_power_training, action_all_with_power = agent.action_all_with_power)   
                    
                    done = False
                    experience = (state.view(1,82),
                                  torch.tensor(action).view(1,21),
                                  torch.tensor(reward).view(1,1),
                                  torch.tensor(next_state).view(1,82),
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
                        
                        soft_update(target = agent.actor_target, source = agent.actor, tau = tau)
                        soft_update(target = agent.critic_target, source = agent.critic, tau = tau)  

                        
                      
        if (step % 20000 == 0) and (step > 0):
            vehicleList = [20,40,60,80,100]
            for numberOfVehicle in vehicleList:
                # testing 
                isTraining = False
                number_of_game = 10
               
                V2I_V2X_Rate_list = np.zeros(number_of_game)
                V2I_Rate_list = np.zeros(number_of_game)
                V2V_Rate_list = np.zeros(number_of_game)
                Fail_percent_list = np.zeros(number_of_game)
           
                for game_idx in range(number_of_game):
                    
                    listOfSelRB = []
                    listOfSelPowerdBm = []
                    
                    env.new_random_game(numberOfVehicle)
                    agent.num_vehicle = numberOfVehicle
                    agent.action_all_with_power = np.zeros([numberOfVehicle, 3, 2],dtype = 'float')
                    test_sample = 200
                    
                    temp_V2V_Rate_list = []
                    temp_V2I_Rate_list = []
                    Rate_list = [] 
                    
                    print('test game idx:', game_idx)
                    for k in range(test_sample):
                        action_temp = agent.action_all_with_power.copy()
                        for i in range(len(env.vehicles)):
                            agent.action_all_with_power[i,:,0] = -1
                            sorted_idx = np.argsort(env.individual_time_limit[i,:])          
                            for j in sorted_idx:                   
                                state = env.get_state(idx = [i,j], isTraining = isTraining, action_all_with_power_training = agent.action_all_with_power_training, action_all_with_power = agent.action_all_with_power)   
                                state = to_tensor(state)    
                                #state를 보고 action을 정함
                                #action은 선택한 power level, 선택한 resource block 정보를 가짐 
                                action = agent.get_action(state).cpu().numpy()
                                         
                                selRBIndex, selPowerdBm = GetRB_Power(fPowermin, action)    
                                #선택한 resource block을 넣어줌
                                agent.action_all_with_power[i, j, 0] = selRBIndex
                                listOfSelRB.append(selRBIndex)
                      
                                #선택한 power level을 넣어줌
                                agent.action_all_with_power[i, j, 1] = selPowerdBm 
                                listOfSelPowerdBm.append(selPowerdBm)
                                    
                            if i % (len(env.vehicles)/10) == 1:
                                action_temp = agent.action_all_with_power.copy()                                
                                returnV2IReward, returnV2VReward, percent = env.act_asyn(action_temp) #self.action_all)            
                                Rate_list.append(np.sum(returnV2IReward) + np.sum(returnV2VReward))
                                temp_V2I_Rate_list.append(np.sum(returnV2IReward))
                                temp_V2V_Rate_list.append(np.sum(returnV2VReward))
                                                        
                    #print("actions", self.action_all_with_power)
                    V2I_V2X_Rate_list[game_idx] = np.mean(np.asarray(Rate_list))
                    V2I_Rate_list[game_idx] = np.mean(np.asarray(temp_V2I_Rate_list))
                    V2V_Rate_list[game_idx] = np.mean(np.asarray(temp_V2V_Rate_list))
                    Fail_percent_list[game_idx] = percent
                    print('failure probability is, ', percent)
                                       
                print ('The number of vehicle is ', len(env.vehicles))
                print ('Mean of the V2I + V2I rate is that ', np.mean(V2I_V2X_Rate_list))
                print ('Mean of the V2I rate is that ', np.mean(V2I_Rate_list))
                print ('Mean of the V2V rate is that ', np.mean(V2V_Rate_list))
                print('Mean of Fail percent is that ', np.mean(Fail_percent_list))      

                savePath = 'ddpg/model/'
                performanceInfo = str(step) + '_' + str(len(env.vehicles))+ '_' + str(np.mean(V2I_Rate_list)) + '_' + str(np.mean(V2V_Rate_list)) + '_' + str(np.mean(Fail_percent_list))       
                criticPath = savePath + 'critic_' + performanceInfo
                actorPath = savePath + 'actor_' + performanceInfo
            
                torch.save(agent.critic_target.state_dict(), criticPath)      
                torch.save(agent.actor_target.state_dict(), actorPath)
        elif (step % 2000 == 0) and (step > 0):
            # testing 
            isTraining = False
            number_of_game = 10
           
            V2I_V2X_Rate_list = np.zeros(number_of_game)
            V2I_Rate_list = np.zeros(number_of_game)
            V2V_Rate_list = np.zeros(number_of_game)
            Fail_percent_list = np.zeros(number_of_game)
            numberOfVehicle = 20
            for game_idx in range(number_of_game):
                
                listOfSelRB = []
                listOfSelPowerdBm = []
                
                env.new_random_game(numberOfVehicle)
                agent.num_vehicle = numberOfVehicle
                agent.action_all_with_power = np.zeros([numberOfVehicle, 3, 2],dtype = 'float')
                
                test_sample = 200
                
                temp_V2V_Rate_list = []
                temp_V2I_Rate_list = []
                Rate_list = [] 
                
                print('test game idx:', game_idx)
                for k in range(test_sample):
                    action_temp = agent.action_all_with_power.copy()
                    for i in range(len(env.vehicles)):
                        agent.action_all_with_power[i,:,0] = -1

                        sorted_idx = np.argsort(env.individual_time_limit[i,:])          
                        for j in sorted_idx:                   
                            state = env.get_state(idx = [i,j], isTraining = isTraining, action_all_with_power_training = agent.action_all_with_power_training, action_all_with_power = agent.action_all_with_power)   
                            state = to_tensor(state)    
                            #state를 보고 action을 정함
                            #action은 선택한 power level, 선택한 resource block 정보를 가짐 
                            action = agent.get_action(state).cpu().numpy()
                                     
                            selRBIndex, selPowerdBm = GetRB_Power(fPowermin, action)    
                            #선택한 resource block을 넣어줌
                            agent.action_all_with_power[i, j, 0] = selRBIndex
                            listOfSelRB.append(selRBIndex)
                  
                            #선택한 power level을 넣어줌
                            agent.action_all_with_power[i, j, 1] = selPowerdBm 
                            listOfSelPowerdBm.append(selPowerdBm)
                                
                        if i % (len(env.vehicles)/10) == 1:
                            action_temp = agent.action_all_with_power.copy()                                
                            returnV2IReward, returnV2VReward, percent = env.act_asyn(action_temp) #self.action_all)            
                            Rate_list.append(np.sum(returnV2IReward) + np.sum(returnV2VReward))
                            temp_V2I_Rate_list.append(np.sum(returnV2IReward))
                            temp_V2V_Rate_list.append(np.sum(returnV2VReward))
                
              
                
                #print("actions", self.action_all_with_power)
                V2I_V2X_Rate_list[game_idx] = np.mean(np.asarray(Rate_list))
                V2I_Rate_list[game_idx] = np.mean(np.asarray(temp_V2I_Rate_list))
                V2V_Rate_list[game_idx] = np.mean(np.asarray(temp_V2V_Rate_list))
                Fail_percent_list[game_idx] = percent
                print('failure probability is, ', percent)
                
            if (isShowTestGraph == 1):
                showSelectHistGraph(listOfSelRB, listOfSelPowerdBm) 
                
            print ('The number of vehicle is ', len(env.vehicles))
            print ('Mean of the V2I + V2I rate is that ', np.mean(V2I_V2X_Rate_list))
            print ('Mean of the V2I rate is that ', np.mean(V2I_Rate_list))
            print ('Mean of the V2V rate is that ', np.mean(V2V_Rate_list))
            print('Mean of Fail percent is that ', np.mean(Fail_percent_list))      

            savePath = 'ddpg/model/'
            performanceInfo = str(step) + '_' + str(np.mean(V2I_Rate_list)) + '_' + str(np.mean(V2V_Rate_list)) + '_' + str(np.mean(Fail_percent_list))       
            criticPath = savePath + 'critic_' + performanceInfo
            actorPath = savePath + 'actor_' + performanceInfo
        
            torch.save(agent.critic_target.state_dict(), criticPath)      
            torch.save(agent.actor_target.state_dict(), actorPath)
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch on TORCS with Multi-modal')

    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
    parser.add_argument('--train_resume', default=1, type=int, help='train resume, using path : ./ddpg/resume model/actor, ./ddpg/resume model/critic')
    parser.add_argument('--using_position_data', default=0, type=int, help='using test data, using path : ./position/vehiclePosition.csv')
    parser.add_argument('--hidden1', default=512, type=int, help='hidden1 num of first fully connect layer')
    parser.add_argument('--hidden2', default=256, type=int, help='hidden2 num of second fully connect layer')
    parser.add_argument('--hidden3', default=128, type=int, help='hidden3 num of first fully connect layer')

    parser.add_argument('--lr_actor', default=0.001, type=float, help='Actor learning rate')
    parser.add_argument('--lr_critic', default=0.0005, type=float, help='Critic learning rate')
    parser.add_argument('--gamma', default=0.99, type=float, help='')   
    parser.add_argument('--bsize', default=256, type=int, help='minibatch size') #256
    parser.add_argument('--rmsize', default=50000, type=int, help='memory size')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')#0.001
    parser.add_argument('--sampling_only_until', default=260, type=int, help='train start step') # 2000

    
    parser.add_argument('--power_min', default=-10.0, type=float, help='') 
    parser.add_argument('--train_iter', default=40000, type=int, help='train iters each timestep')
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
    nVeh = 20

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
    

    if args.mode =='train':
        if args.train_resume == 1:
            agent.actor.load_state_dict(torch.load('./ddpg/resume model/actor'))        
            agent.actor_target.load_state_dict(torch.load('./ddpg/resume model/actor'))
            agent.critic.load_state_dict(torch.load('./ddpg/resume model/critic'))
            agent.critic_target.load_state_dict(torch.load('./ddpg/resume model/critic'))
            
        train(args, memory, agent, env,args.power_min)
     

    
