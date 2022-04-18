#!/usr/bin/env python3 
import numpy as np
import argparse
from copy import deepcopy
import torch
import gym
import matplotlib.pyplot as plt

from Environment import *
from normalized_env import NormalizedEnv
from evaluator import Evaluator
from ddpg import DDPG
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
    action[20] = ((action[20] + 1.0) * (powerRange / 2)) + powerMin
    action[20] = np.clip(action[20], powerMin, 23.0)
    selectedPower = action[20]
    
    return selectedRBIndex, selectedPower

def train(num_iterations, agent, env,  evaluate, args):
    
    isShowTestGraph = args.showtestGraph
    validate_steps = args.validate_steps
    output = args.output
    max_episode_length = args.max_episode_length
    batchsize= args.bsize
    debug=args.debug
    minPower = args.power_min
    agent.is_training = True
    step = 0
    observation = None   
    
    selectStep = 0   
    
    listOfselRB = []
    listOfselPower = []
    listOfReward = []
    
    listofState_nextState = []
    
    reward_sum = 0
    reward_sum_list = []
    
    while step < num_iterations:
        
        agent.is_training = True
        
        if step % 2000 == 1:
            observation = None
            env.new_random_game(env.n_Veh)  
            print('episode : ' + str(int(step / 2000)) + ' reward total : ' + str(reward_sum))
            
        print(step)           
        # reset if it is the start of episode
        if observation is None:
            reward_sum_list.append(reward_sum)
            reward_sum = 0
            env.new_random_game()
            state = env.get_state(idx = [0,0], isTraining = False, action_all_with_power_training = agent.action_all_with_power_training, action_all_with_power = agent.action_all_with_power)
            observation = deepcopy(state)
            agent.reset(observation)
           
        #현재 상태에서 모든 차량이 선택을 수행함.
        for k in range(1):
            #i번째 송신 차량
            for i in range(len(env.vehicles)):
                
                
                for j in range(0,3): 
                    
                    # i번째 차량에서 j번째 차량으로 데이터를 전송할때 state를 가져옴.
                    #i번째 송신 차량에서 전송한 신호를 수신하는 j번째 수신 차량 
                    observation = env.get_state(idx = [i,j], isTraining = False, action_all_with_power_training = agent.action_all_with_power_training, action_all_with_power = agent.action_all_with_power)                    
                    
                    observation = deepcopy(observation)                     
                    agent.reset_state(observation)
                    
                    #state를 보고 action을 정함
                    #action은 선택한 power level, 선택한 resource block 정보를 가짐
                    # 랜덤 선택
                    if selectStep <= args.warmup:
                        print('random')
                        action = agent.random_action()
                    else:
                        print('policy')
                        action = agent.select_action(observation, decay_epsilon=True)
                        
                    
                    selRBIndex, selPowerdBm = GetRB_Power(minPower,action)
        
                    print('sel RB, PowerdBm : ', [selRBIndex, selPowerdBm] )
                    
                    selectStep += 1
                                                                                     
                    #i 번째 차량에서 j 번째 차량으로 전송할 리소스 블럭 선택
                    agent.action_all_with_power_training[i, j, 0] = selRBIndex  # 선택한 Resourceblock을 저장함. 
                        
                    #i 번째 차량에서 j 번째 차량으로 전송할 Power dB 선택
                    agent.action_all_with_power_training[i, j, 1] = selPowerdBm # PowerdBm을 저장함.
                                        
                    
                    #선택한 power level과 resource block을 기반으로 reward를 계산함.
                    reward = env.act_for_training(agent.action_all_with_power_training, [i,j]) 
                    
                    listOfselRB.append(selRBIndex)
                    listOfselPower.append(selPowerdBm)
                    listOfReward.append(reward)
    
                    reward_sum += reward
                    
                    observation2 = env.get_state(idx = [i,j], isTraining = False, action_all_with_power_training = agent.action_all_with_power_training, action_all_with_power = agent.action_all_with_power) 
                    
                    listofState_nextState.append(np.concatenate((deepcopy(observation), deepcopy(observation2)), axis=0))
                    
                    done = False                    
                    info = False
                    
                    observation2 = deepcopy(observation2)
                       
                    # agent observe and update policy
                    agent.observe(reward, observation2, done)
                    
                    if selectStep > args.warmup and selectStep > batchsize:
                        agent.update_policy()

                    #observation = deepcopy(observation2)
        
        step += 1
        
       
        
        if (step % validate_steps == 0) and (step > 0):
                # testing 
                agent.is_training = False                
                number_of_game = 10
                if (step % 10000 == 0) and (step > 0):
                    number_of_game = 50 
                if (step == 38000):
                    number_of_game = 100               
                V2I_Rate_list = np.zeros(number_of_game)
                V2V_Rate_list = np.zeros(number_of_game)
                Fail_percent_list = np.zeros(number_of_game)
                for game_idx in range(number_of_game):
                    env.new_random_game(env.n_Veh)
                    test_sample = 200
                    V2IRate_list = []
                    V2VRate_list = []                    
                    selPowerRateList = []
                    selRBRateList = []
                    
                    print('test game idx:', game_idx)
                    for k in range(test_sample):
                        action_temp = agent.action_all_with_power.copy()
                        for i in range(len(env.vehicles)):
                            agent.action_all_with_power[i,:,0] = -1
                            sorted_idx = np.argsort(env.individual_time_limit[i,:])          
                            for j in sorted_idx:                   
                                observation = env.get_state(idx = [i,j], isTraining = True, action_all_with_power_training = agent.action_all_with_power_training, action_all_with_power = agent.action_all_with_power)                           
                                action = agent.select_action(observation, decay_epsilon=False)
                                
                                selRBIndex, selPowerdBm = GetRB_Power(action)
   
                                selRBRateList.append(selRBIndex)
                                selPowerRateList.append(selPowerdBm)
                                
                                agent.action_all_with_power[i, j, 0] = selRBIndex
                                agent.action_all_with_power[i, j, 1] = selPowerdBm

                            if i % (len(env.vehicles)/10) == 1:
                                action_temp = agent.action_all_with_power.copy()
                                V2IRate, V2VRate, percent = env.act_asyn(action_temp) #self.action_all)            
                                V2IRate_list.append(np.sum(V2IRate))
                                V2VRate_list.append(np.sum(V2VRate))
                    
                    if (isShowTestGraph == 1):
                        plt.subplot(211)
                        plt.hist(selRBRateList, histtype='step')
                        plt.title('Left : RB Rate, Right : Power Rate')
                        plt.subplot(212)
                        plt.hist(selPowerRateList, histtype='step')
                        plt.show()
                    
                    V2I_Rate_list[game_idx] = np.mean(np.asarray(V2IRate_list))
                    V2V_Rate_list[game_idx] = np.mean(np.asarray(V2VRate_list))
                    Fail_percent_list[game_idx] = percent
                    print('failure probability is, ', percent)

                    
                #테스트 할때마다 모델 저장
                performanceInfo = str(step) + '_' + str(np.mean(V2I_Rate_list)) + '_' + str(np.mean(V2V_Rate_list)) + '_' + str(np.mean(Fail_percent_list))               
                agent.save_model('ddpg/model', performanceInfo)
                print ('The number of vehicle is ', len(env.vehicles))
                print ('Mean of the V2I rate + V2V rate is that ', np.mean(V2I_Rate_list) + np.mean(V2V_Rate_list))
                print ('Mean of the V2I rate is that ', np.mean(V2I_Rate_list))
                print ('Mean of the V2V rate is that ', np.mean(V2V_Rate_list))
                print('Mean of Fail percent is that ', np.mean(Fail_percent_list))   
     
            
    agent.save_model('ddpg/model', 'final')
    

    totalModelPath = 'ddpg/'
    all_select_rb = np.reshape(listOfselRB,(len(listOfselRB),1))
    all_select_power = np.reshape(listOfselPower,(len(listOfselPower),1))
    listOfReward = np.reshape(listOfReward,(len(listOfReward),1))
    
    listofLog = np.concatenate((all_select_rb, all_select_power,listOfReward), axis=1)
    
    MakeCSVFile(totalModelPath, 'total_log.csv', ['Select_RB','Select_Power','Reward'], listofLog)

    listOfStateLog =  np.array(listofState_nextState)
    MakeCSVFile(totalModelPath, 'State_transition_log.csv', ['current_state : 82', ' next_state : 82'], listOfStateLog)
    
    return reward_sum_list

def test(num_episodes, agent, env, evaluate, model_path, visualize=True, debug=False):

    agent.load_weights(model_path)
    agent.is_training = False
    agent.eval()
    policy = lambda x: agent.select_action(x, decay_epsilon=False)

    for i in range(num_episodes):
        validate_reward = evaluate(env, policy, debug=debug, visualize=visualize, save=False)
        if debug: prYellow('[Evaluate] #{}: mean_reward:{}'.format(i, validate_reward))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch on TORCS with Multi-modal')

    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
    parser.add_argument('--hidden1', default=512, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2', default=256, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--prate', default=0.0001, type=float, help='policy net learning rate (only for DDPG)')
    
    parser.add_argument('--discount', default=0.99, type=float, help='')
    parser.add_argument('--bsize', default=256, type=int, help='minibatch size') # 64
    parser.add_argument('--rmsize', default=6000000, type=int, help='memory size')
    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--ou_theta', default=0.15, type=float, help='noise theta')
    parser.add_argument('--ou_sigma', default=0.2, type=float, help='noise sigma') 
    parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu') 
    parser.add_argument('--validate_episodes', default=20, type=int, help='how many episode to perform during validate experiment')
    parser.add_argument('--max_episode_length', default=500, type=int, help='')

    parser.add_argument('--output', default='output', type=str, help='')
    parser.add_argument('--debug', dest='debug', action='store_true')
    
    parser.add_argument('--power_min', default=-10.0, type=float, help='') 
    parser.add_argument('--init_w', default=0.003, type=float, help='') 
    parser.add_argument('--warmup', default=4000, type=int, help='time without training but only filling the replay memory') # 10000
    parser.add_argument('--validate_steps', default=2000, type=int, help='how many steps to perform a validate experiment') # 2000
    parser.add_argument('--train_iter', default=4001, type=int, help='train iters each timestep') # 200000
    parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--seed', default=-1, type=int, help='')
    parser.add_argument('--resume', default='default', type=str, help='Resuming model path for testing')
    parser.add_argument('--showtestGraph', default='0', type=int, help='Show graph for test')
    # parser.add_argument('--l2norm', default=0.01, type=float, help='l2 weight decay') # TODO
    # parser.add_argument('--cuda', dest='cuda', action='store_true') # TODO

    args = parser.parse_args()
   # args.output = get_output_folder(args.output, args.env)
  #  if args.resume == 'default':
  #      args.resume = 'output/{}-run0'.format(args.env)
        
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

    if args.seed > 0:
        np.random.seed(args.seed)
        env.seed(args.seed)

    nb_states = env.observation_space.shape[0]
    nb_actions = env.action_space.shape[0]

    agent = DDPG(nb_states, nb_actions, args, nVeh)
    
    evaluate = Evaluator(args.validate_episodes, 
        args.validate_steps, args.output, max_episode_length=args.max_episode_length)

    if args.mode == 'train':
        train(args.train_iter, agent, env, evaluate, 
            args)

    elif args.mode == 'test':
        test(args.validate_episodes, agent, env, evaluate, args.resume,
            visualize=True, debug=args.debug)

    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))
