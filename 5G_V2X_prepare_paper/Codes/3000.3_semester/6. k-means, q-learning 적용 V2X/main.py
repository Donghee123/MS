import random


from Environment import *

import os
import csv
import torch
import wandb 
import pandas as pd
import argparse
from sequential_k_means import *

sumrateV2IList = []
sumrateV2VList = []

probabilityOfSatisfiedV2VList = []

#def train(env, agent, memory, batch_size, train_iter, wandb):
def train(env, train_iter):
    ep_reward = 0.
    
    
    ep_reward = []        
    env.new_random_game(20)
        
    eps_max = 0.08

    c = []

    logs = []

    state_1 = env.get_state([0,0], True) 
    state_2 = env.get_state([1,0], True) 

    distance = euclideanDistance
    points = [state_1, state_2]
    numPoints = len(points)
    k = numPoints
    # Use K-Mean++ to pick k distant starting centroids.

    clusters = [Cluster([c]) for c in kmeanpp(points, k, distance)]

    for step in (range(0, train_iter)): # need more configuration
        if step == 0:                   # initialize set some varibles
            ep_reward, ep_target_reward = 0., 0.
            
        
        if (step % 2000 == 1):
            env.new_random_game(20)
  
            
        print(step)

    

        for k in range(1):
            #i번째 송신 차량
            for i in range(len(env.vehicles)):
                #i번째 송신 차량에서 전송한 신호를 수신하는 j번째 수신 차량 
                for j in range(3): 
                    # i번째 차량에서 j번째 차량으로 데이터를 전송할때 state를 가져옴.
                
                    state_old = env.get_state([i,j], True)
                    
      
                    #k-means clustring에 추가할 부분
                    state_old_kmeans = state_old

                    #state를 보고 action을 정함
                    #action은 선택한 power level, 선택한 resource block 정보를 가짐
                    #action = agent.get_action(state_old_kmeans)                    
                    #self.merge_action([i,j], action)   
                    
                    #선택한 resource block을 넣어줌
                    #agent.action_all_with_power_training[i, j, 0] = action % agent.RB_number 
                    
                    #선택한 power level을 넣어줌
                    #agent.action_all_with_power_training[i, j, 1] = int(np.floor(action/agent.RB_number))  
                                     
                    #선택한 power level과 resource block을 기반으로 reward를 계산함.
                    reward_train, reward_best = env.act_for_training(env.action_all_with_power_training, [i,j])
                    
                    ep_reward = ep_reward + reward_train
                    ep_target_reward = ep_target_reward + reward_best

                    
                    state_new = env.get_state([i,j], True) 
                    
                    experience = (state_old, 
                                  torch.tensor(action).view(1,1),
                                  torch.tensor(reward_train).view(1,1),
                                  torch.tensor(state_new).view(1,82),
                                  torch.tensor(np.array([0]).reshape(1,1)))
                    #경험 저장
                    memory.push(experience)
                    
                    #메모리의 값이 충분히 찼다면 업데이트를 시킴
                    if len(memory) >= batch_size:
                        sampled_exps = memory.sample(batch_size)
                        sampled_exps = prepare_training_inputs(sampled_exps)
                        sampled_exps = sampled_exps
                        agent.update(*sampled_exps)
                    
                    #qnet의 파라미터를 하이퍼 파라미터 수만큼 업데이트 했다면 target qnet에 qnet의 파라미터를 업데이트함
                    if step % 50 == 0:
                        agent.qnet_target.load_state_dict(agent.qnet.state_dict())
        
        logs.append((": 2000 step Cumulative Reward : " + str(ep_reward) + '-- Target Reward : ' + str(ep_target_reward) + '-- Diff DQN vs target : ' + str(ep_target_reward - ep_reward)))
        wandb.log({"DQN_reward": ep_reward, "target_reward" : ep_target_reward, "Diff DQN vs target" : ep_target_reward - ep_reward})
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
                env.new_random_game(agent.num_vehicle)
                test_sample = 200
                Rate_list = []
                temp_V2V_Rate_list = []
                temp_V2I_Rate_list = []
                print('test game idx:', game_idx)
                for k in range(test_sample):
                    action_temp = agent.action_all_with_power.copy()
                    for i in range(len(env.vehicles)):
                        agent.action_all_with_power[i,:,0] = -1
                        sorted_idx = np.argsort(env.individual_time_limit[i,:])          
                        for j in sorted_idx:                   
                            state_old = env.get_state([i,j], False, agent.action_all_with_power_training, agent.action_all_with_power) 

                            state_old = torch.tensor(state_old).view(1,82).float()
                            
                            action = agent.get_action(state_old.to(agent.DEVICE),isTest = True)
                            selectedRB = action % agent.RB_number
                            resourceBlocks[selectedRB][0] = resourceBlocks[selectedRB][0] + 1
                            agent.action_all_with_power[i, j, 0] = action % agent.RB_number
                            agent.action_all_with_power[i, j, 1] = int(np.floor(action/agent.RB_number))
                            
                            
                        if i % (len(env.vehicles)/10) == 1:
                            action_temp = agent.action_all_with_power.copy()                                
                            returnV2IReward, returnV2VReward, percent = env.act_asyn(action_temp) #self.action_all)            
                            Rate_list.append(np.sum(returnV2IReward) + np.sum(returnV2VReward))
                            temp_V2I_Rate_list.append(np.sum(returnV2IReward))
                            temp_V2V_Rate_list.append(np.sum(returnV2VReward))
                            
                V2I_V2X_Rate_list[game_idx] = np.mean(np.asarray(Rate_list))
                V2I_Rate_list[game_idx] = np.mean(np.asarray(temp_V2I_Rate_list))
                V2V_Rate_list[game_idx] = np.mean(np.asarray(temp_V2V_Rate_list))
                Fail_percent_list[game_idx] = percent
                #print("action is", self.action_all_with_power)
                print('failure probability is, ', percent)
                #print('action is that', action_temp[0,:])
            
            histTable = wandb.Table(data=resourceBlocks, columns=["Resource block"])
            wandb.log({'my_histogram': wandb.plot.histogram(histTable, "Resource block",title="Count of Resource block")})   
            print ('The number of vehicle is ', len(env.vehicles))
            print ('Mean of the V2I + V2I rate is that ', np.mean(V2I_V2X_Rate_list))
            print ('Mean of the V2I rate is that ', np.mean(V2I_Rate_list))
            print ('Mean of the V2V rate is that ', np.mean(V2V_Rate_list))
            print('Mean of Fail percent is that ', np.mean(Fail_percent_list))                   
            #print('Test Reward is ', np.mean(test_result))
            
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

    states = torch.cat(states, dim=0).float().to(device)
    actions = torch.cat(actions, dim=0).to(device)
    rewards = torch.cat(rewards, dim=0).float().to(device)
    next_states = torch.cat(next_states, dim=0).float().to(device)
    dones = torch.cat(dones, dim=0).float().to(device)
    return states, actions, rewards, next_states, dones

#File 유틸 함수들    
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
     
def MakeCSVFile(strFolderPath, strFilePath, aryOfDatas):
    strTotalPath = "%s\%s" % (strFolderPath,strFilePath)
    
    f = open(strTotalPath,'w', newline='')
    wr = csv.writer(f)
    wr.writerow(["V2I sumrate", "V2V sumrate", "V2I, V2V sumrate", "outageprobability"])
    
    for i in range(0,len(aryOfDatas)):
        wr.writerow(aryOfDatas[i])
    
    f.close()

def main(args):

  #wandb.init(config=args, project="my-project")
  #wandb.config["My pytorch DQN"] = "DQN Version 0.1"

  up_lanes = [3.5/2,3.5/2 + 3.5,250+3.5/2, 250+3.5+3.5/2, 500+3.5/2, 500+3.5+3.5/2]
  down_lanes = [250-3.5-3.5/2,250-3.5/2,500-3.5-3.5/2,500-3.5/2,750-3.5-3.5/2,750-3.5/2]
  left_lanes = [3.5/2,3.5/2 + 3.5,433+3.5/2, 433+3.5+3.5/2, 866+3.5/2, 866+3.5+3.5/2]
  right_lanes = [433-3.5-3.5/2,433-3.5/2,866-3.5-3.5/2,866-3.5/2,1299-3.5-3.5/2,1299-3.5/2]
  arrayOfVeh = [40] # for train
  
  width = 750
  height = 1299

  
  Env = Environ(down_lanes,up_lanes,left_lanes,right_lanes, width, height, arrayOfVeh[0])
  
  
  n_output = 60
  lr = args.learning_rate
  gamma = args.gamma
 
  
  train_iter = args.train_iter
  
  for nVeh in arrayOfVeh:      
      
      Env.new_random_game()
         
      #학습
      #train(Env, train_iter, wandb)
      train(Env, train_iter)

      #학습 전
      #v2i_Sumrate, v2v_Sumrate, probability = agent.play(n_step = 100, n_episode = 20, random_choice = False)
        
      #sumrateV2IList.append(v2i_Sumrate)
      #sumrateV2VList.append(v2v_Sumrate)
        
      #probabilityOfSatisfiedV2VList.append(probability)
        
        
      #학습 후
      #agent.play()

  sumrateV2IListnpList = np.array(sumrateV2IList)
  sumrateV2VListnpList = np.array(sumrateV2VList)
  sumrateV2V_V2IListnpList = sumrateV2IListnpList + sumrateV2VListnpList
  probabilityOfSatisfiedV2VnpList = np.array(probabilityOfSatisfiedV2VList)
  
  print('V2I sumrate')
  print(sumrateV2IListnpList)
  print('V2V sumrate')
  print(sumrateV2VListnpList)
  print('V2V + V2I rate')
  print(sumrateV2IListnpList + sumrateV2VListnpList)
  print('Outage probability')
  print(probabilityOfSatisfiedV2VnpList)

  allData=[]
  allData.append(sumrateV2IListnpList)
  allData.append(sumrateV2VListnpList)
  allData.append(sumrateV2V_V2IListnpList)
  allData.append(probabilityOfSatisfiedV2VnpList)
  allData = np.transpose(allData)
  
  folderPath = './ResultData'
  csvFileName = 'ResultData.csv'
  
  createFolder(folderPath)
  MakeCSVFile(folderPath, csvFileName, allData)
  
  
if __name__ == '__main__':
    

    parser = argparse.ArgumentParser(description='PyTorch on DQN')

    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
    parser.add_argument('--train_resume', default=1, type=int, help='train resume, using path : ./ddpg/resume model/actor, ./ddpg/resume model/critic')
    parser.add_argument('--hidden1', default=128, type=int, help='hidden1 num of first fully connect layer')
    parser.add_argument('--hidden2', default=256, type=int, help='hidden2 num of second fully connect layer')
    parser.add_argument('--hidden3', default=120, type=int, help='hidden3 num of first fully connect layer')
    parser.add_argument('--learning_rate', default=0.0001, type=float, help='learning rate')   
    parser.add_argument('--gamma', default=0.98, type=float, help='reward gamma')   
    parser.add_argument('--memorysize', default=1000000, type=int, help='memory size')
    parser.add_argument('--batchszie', default=50, type=int, help='batch size')
    parser.add_argument('--train_iter', default=40000, type=int, help='train iters each timestep')

    args = parser.parse_args()
    
    main(args)
