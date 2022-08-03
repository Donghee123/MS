from __future__ import division, print_function
import random
#import tensorflow.compat.v1 as tf
import tensorflow as tf
from agent import Agent
from Environment import *
import pandas as pd
import csv
import os
import pdb
from utils import *

flags = tf.app.flags


  
# Model
flags.DEFINE_string('model', 'm1', 'Type of model')
flags.DEFINE_boolean('dueling', False, 'Whether to use dueling deep q-network')
flags.DEFINE_boolean('double_q', False, 'Whether to use double q-learning')

# Etc
flags.DEFINE_boolean('use_gpu', True, 'Whether to use gpu or not')
flags.DEFINE_string('gpu_fraction', '1/1', 'idx / # of gpu fraction e.g. 1/3, 2/3, 3/3')
flags.DEFINE_integer('random_seed', 123, 'Value of random seed')


FLAGS = flags.FLAGS

# Set random seed
tf.set_random_seed(FLAGS.random_seed)
random.seed(FLAGS.random_seed)

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

def MakeTime_PowerCSVFile(strFolderPath, strFilePath, selcted_prob_23dBmnpList, selcted_prob_10dBmnpList, selcted_prob_5dBmnpList):
    strTotalPath = "%s\%s" % (strFolderPath,strFilePath)
    
    for i in range(len(selcted_prob_23dBmnpList)): #차량의 수 : 20, 40, 60, 80, 100 에 대한 인덱스
      for ii in range(len(selcted_prob_23dBmnpList[i])): #에피소드 수 : 보통 20개로 함.

        f = open(f'{strTotalPath}_{arrayOfVeh[i]}_{ii}','w', newline='')
        wr = csv.writer(f)
        wr.writerow(["remain time, selcted prob 23dBm", "selcted prob 10dBm", "selcted prob 5dBm"])

        for iii in range(len(selcted_prob_23dBmnpList[i][ii][0])):
          wr.writerow([selcted_prob_23dBmnpList[i][ii][0][iii], selcted_prob_23dBmnpList[i][ii][1][iii], selcted_prob_10dBmnpList[i][ii][1][iii], selcted_prob_5dBmnpList[i][ii][1][iii]])
      
        f.close()

if FLAGS.gpu_fraction == '':
  raise ValueError("--gpu_fraction should be defined")

def calc_gpu_fraction(fraction_string):
  idx, num = fraction_string.split('/')
  idx, num = float(idx), float(num)

  fraction = 1 / (num - idx + 1)
  print(" [*] GPU : %.4f" % fraction)
  return fraction

def play_multi_agent(agents, env, n_step, n_episode, use_async):

  use_async = use_async
  number_of_game = n_episode

  V2I_Rate_list = [np.zeros(number_of_game) for _ in list(AgentMode)]
  V2V_Rate_list = [np.zeros(number_of_game) for _ in list(AgentMode)]
  power_list =  [np.zeros(number_of_game) for _ in list(AgentMode)]
  varpower_list = [np.zeros(number_of_game) for _ in list(AgentMode)]
  stdpower_list = [np.zeros(number_of_game) for _ in list(AgentMode)]
  V2V_AvGRate_list = [np.zeros(number_of_game) for _ in list(AgentMode)]
  Fail_percent_list = [np.zeros(number_of_game) for _ in list(AgentMode)]

  selcted_prob_23dBm = [np.zeros(number_of_game) for _ in list(AgentMode)]
  selcted_prob_10dBm = [np.zeros(number_of_game) for _ in list(AgentMode)]
  selcted_prob_5dBm = [np.zeros(number_of_game) for _ in list(AgentMode)]

       
  for game_idx in range(number_of_game):
      env.new_random_game(agents[0].num_vehicle)
      test_sample = n_step

      Rate_list = [[] for _ in list(AgentMode)]
      Rate_list_V2V = [[] for _ in list(AgentMode)]
      Rate_list_V2VAvg = [[] for _ in list(AgentMode)]

      print('test game idx:', game_idx)
      print('The number of vehicle is ', len(env.vehicles))
            
      time_left_list = [[] for _ in list(AgentMode)]
      power_select_list_0 = [[] for _ in list(AgentMode)]
      power_select_list_1 = [[] for _ in list(AgentMode)]
      power_select_list_2 = [[] for _ in list(AgentMode)]

      for agentindex, agent in enumerate(agents):
        for k in range(test_sample):
          action_temp = agent.action_all_with_power.copy()

          for i in range(len(env.vehicles)):
            agent.action_all_with_power[i, :, 0] = -1
            sorted_idx = np.argsort(env.individual_time_limit[agentindex][i, :])
            for j in sorted_idx:
              state_old = agent.get_state_external_env(agentindex, env, [i, j])
              time_left_list[agentindex].append(state_old[-1])
              action = agent.predictwidthModel(agent, state_old)
                          
              if state_old[-1] <=0:
                continue
                          
              power_selection = int(np.floor(action/agent.RB_number))
                          
              if power_selection == 0:
                power_select_list_0[agentindex].append(state_old[-1])
              if power_selection == 1:
                power_select_list_1[agentindex].append(state_old[-1])
              if power_selection == 2:
                power_select_list_2[agentindex].append(state_old[-1])
                          
              agent.merge_action([i, j], action)
                      
              #시뮬레이션 차량의 갯수 / 10 만큼 action이 정해지면 act를 수행함.
              if (i % (len(env.vehicles) / 10) == 1) & (use_async is True):
                action_temp = agent.action_all_with_power.copy()
                rewardOfV2I, rewardOfV2V, percent = env.act_asyn(action_temp) 
                
                Rate_list[agentindex].append(np.sum(rewardOfV2I))
                Rate_list_V2V[agentindex].append(np.sum(rewardOfV2V))
                Rate_list_V2VAvg[agentindex].append(np.mean(rewardOfV2V))

          if use_async is False:
            action_temp = agent.action_all_with_power.copy()
            rewardOfV2I, rewardOfV2V, percent = env.act(action_temp) 
            Rate_list[agentindex].append(np.sum(rewardOfV2I))
            Rate_list_V2V[agentindex].append(np.sum(rewardOfV2V))
            Rate_list_V2VAvg[agentindex].append(np.mean(rewardOfV2V))  

            
          selected_power = [23 for _ in range(len(power_select_list_0))] + [10 for _ in range(len(power_select_list_1))] + [5 for _ in range(len(power_select_list_2))]
              
          number_0, bin_edges = np.histogram(power_select_list_0, bins = 10)

          number_1, bin_edges = np.histogram(power_select_list_1, bins = 10)

          number_2, bin_edges = np.histogram(power_select_list_2, bins = 10)


          p_0 = number_0 / (number_0 + number_1 + number_2)
          p_1 = number_1 / (number_0 + number_1 + number_2)
          p_2 = number_2 / (number_0 + number_1 + number_2)

             
          V2I_Rate_list[agentindex][game_idx] = np.mean(np.asarray(Rate_list[agentindex]))
          V2V_Rate_list[agentindex][game_idx] = np.mean(np.asarray(Rate_list_V2V[agentindex]))
          V2V_AvGRate_list[agentindex][game_idx] = np.mean(np.asarray(Rate_list_V2VAvg[agentindex]))

          Fail_percent_list[agentindex][game_idx] = percent

          power_list[agentindex][game_idx] = np.mean(selected_power)
          varpower_list[agentindex][game_idx] = np.var(selected_power)
          stdpower_list[agentindex][game_idx] = np.std(selected_power)
          selcted_prob_23dBm[agentindex].append([bin_edges[:-1]*0.1 + 0.01, p_0])
          selcted_prob_10dBm[agentindex].append([bin_edges[:-1]*0.1 + 0.01, p_1])
          selcted_prob_5dBm[agentindex].append([bin_edges[:-1]*0.1 + 0.01, p_2])


          print('Mean of the V2I rate is that ', np.mean(V2I_Rate_list[agentindex][0:game_idx] ))
          print('Mean of the V2V rate is that ', np.mean(V2V_Rate_list[agentindex][0:game_idx] ))
          print('Mean of the patital V2V Avg rate is that ', np.mean(V2V_AvGRate_list[agentindex][0:game_idx] ))
          print('Mean of Fail percent is that ',percent, np.mean(Fail_percent_list[agentindex][0:game_idx]))
          print('Mean of Selected Power ',np.mean(power_list[agentindex][0:game_idx]))
          print('Mean of Selected var Power ',np.mean(varpower_list[agentindex][0:game_idx]))
          print('Mean of Selected std Power ',np.mean(stdpower_list[agentindex][0:game_idx]))


        print('The number of vehicle is ', len(env.vehicles))
        print('Mean of the V2I rate is that ', np.mean(V2I_Rate_list[agentindex]))
        print('Mean of the V2V rate is that ', np.mean(V2V_Rate_list[agentindex]))
        print('Mean of the V2V avg rate is that ', np.mean(V2V_AvGRate_list[agentindex]))
          
        print('Mean of Fail percent is that ', np.mean(Fail_percent_list[agentindex]))
        print('Mean of Selected Power ',np.mean(power_list[agentindex]))
        print('Mean of Selected var Power ',np.mean(varpower_list[agentindex]))
        print('Mean of Selected std Power ',np.mean(stdpower_list[agentindex]))
  
  return V2I_Rate_list, V2V_Rate_list, V2V_AvGRate_list, Fail_percent_list, power_list, varpower_list, stdpower_list, selcted_prob_23dBm, selcted_prob_10dBm, selcted_prob_5dBm
  
arrayOfVeh = [20]#, 40, 60, 80, 100] # for play
  #arrayOfVeh = [20] # for train

def main(_):
  use_async = False
  up_lanes = [3.5/2,3.5/2 + 3.5,250+3.5/2, 250+3.5+3.5/2, 500+3.5/2, 500+3.5+3.5/2]
  down_lanes = [250-3.5-3.5/2,250-3.5/2,500-3.5-3.5/2,500-3.5/2,750-3.5-3.5/2,750-3.5/2]
  left_lanes = [3.5/2,3.5/2 + 3.5,433+3.5/2, 433+3.5+3.5/2, 866+3.5/2, 866+3.5+3.5/2]
  right_lanes = [433-3.5-3.5/2,433-3.5/2,866-3.5-3.5/2,866-3.5/2,1299-3.5-3.5/2,1299-3.5/2]
  
  width = 750
  height = 1299
  

  sumrateV2IList = []
  sumrateV2VList = []

  probabilityOfSatisfiedV2VList = []
  energyeffcientList = []
  varpowerList = []
  stdpowerList = []

  selcted_prob_23dBmList = []
  selcted_prob_10dBmList = []
  selcted_prob_5dBmList = []

  for nVeh in arrayOfVeh:      
      Env = Environ(down_lanes,up_lanes,left_lanes,right_lanes, width, height,nVeh)
      Env.new_random_game()
      gpu_options = tf.GPUOptions(
          per_process_gpu_memory_fraction=calc_gpu_fraction(FLAGS.gpu_fraction))
      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True
    
      with tf.Session(config=config) as sess:
        config = []
        agent = Agent(config, Env, sess)
        agent.training = False

        #학습 전
        v2i_Sumrate, v2v_Sumrate, probability, powersum, varpower, stdpower, selcted_prob_23dBm, selcted_prob_10dBm, selcted_prob_5dBm = agent.playwithKeras(n_step = 200, n_episode = 20, random_choice = False, use_async = use_async)
        
        selcted_prob_23dBmList.append(selcted_prob_23dBm)
        selcted_prob_10dBmList.append(selcted_prob_10dBm)
        selcted_prob_5dBmList.append(selcted_prob_5dBm)

        sumrateV2IList.append(v2i_Sumrate)
        sumrateV2VList.append(v2v_Sumrate)
        probabilityOfSatisfiedV2VList.append(probability)
        energyeffcientList.append(powersum)
        varpowerList.append(varpower)
        stdpowerList.append(stdpower)

  sumrateV2IListnpList = np.array(sumrateV2IList)
  sumrateV2VListnpList = np.array(sumrateV2VList)
  sumrateV2V_V2IListnpList = sumrateV2IListnpList + sumrateV2VListnpList
  probabilityOfSatisfiedV2VnpList = np.array(probabilityOfSatisfiedV2VList)
  meanpowernpList = np.array(energyeffcientList)
  varpowernpList = np.array(varpowerList)
  stdpowernpList = np.array(stdpowerList)
  selcted_prob_23dBmnpList = np.array(selcted_prob_23dBmList)
  selcted_prob_10dBmnpList = np.array(selcted_prob_10dBmList)
  selcted_prob_5dBmnpList = np.array(selcted_prob_5dBmList)

  print('V2I sumrate')
  print(sumrateV2IListnpList)
  print('V2V sumrate')
  print(sumrateV2VListnpList)
  print('V2V + V2I rate')
  print(sumrateV2IListnpList + sumrateV2VListnpList)
  print('Outage probability')
  print(probabilityOfSatisfiedV2VnpList)

  print('mean power')
  print(meanpowernpList)
  print('var power')
  print(varpowernpList)
  print('std power')
  print(stdpowernpList)

  allData=[]
  allData.append(sumrateV2IListnpList)
  allData.append(sumrateV2VListnpList)
  allData.append(sumrateV2V_V2IListnpList)
  allData.append(probabilityOfSatisfiedV2VnpList)
  allData.append(meanpowernpList)
  allData.append(varpowernpList)
  allData.append(stdpowernpList)

  allData = np.transpose(allData)
  
  folderPath = './ResultData'
  csvFileName = 'ResultData.csv'
  csvtime_powerName = 'Result_time_power_Data.csv'

  createFolder(folderPath)
  MakeCSVFile(folderPath, csvFileName, allData)

  MakeTime_PowerCSVFile(folderPath, csvtime_powerName, selcted_prob_23dBmnpList, selcted_prob_10dBmnpList, selcted_prob_5dBmnpList)
  
if __name__ == '__main__':
    tf.app.run()
