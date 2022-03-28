import random

from Environment import *

import os
import pandas as pd
from pandas import Series, DataFrame
import tensorflow as tf
from agent import Agent

flags = tf.app.flags

sumrateV2IList = []
sumrateV2VList = []

probabilityOfSatisfiedV2VList = []
  
# Model
flags.DEFINE_string('model', 'm1', 'Type of model')
flags.DEFINE_boolean('dueling', False, 'Whether to use dueling deep q-network')
flags.DEFINE_boolean('double_q', False, 'Whether to use double q-learning')

# Environment
flags.DEFINE_string('env_name', 'Breakout-v0', 'The name of gym environment to use')
flags.DEFINE_integer('action_repeat', 4, 'The number of action to be repeated')

# Etc
flags.DEFINE_boolean('use_gpu', True, 'Whether to use gpu or not')
flags.DEFINE_string('gpu_fraction', '1/1', 'idx / # of gpu fraction e.g. 1/3, 2/3, 3/3')
flags.DEFINE_boolean('display', False, 'Whether to do display the game screen or not')
flags.DEFINE_boolean('is_train', True, 'Whether to do training or testing')
flags.DEFINE_integer('random_seed', 123, 'Value of random seed')

FLAGS = flags.FLAGS

# Set random seed
tf.set_random_seed(FLAGS.random_seed)
random.seed(FLAGS.random_seed)

if FLAGS.gpu_fraction == '':
  raise ValueError("--gpu_fraction should be defined")

def calc_gpu_fraction(fraction_string):
  idx, num = fraction_string.split('/')
  idx, num = float(idx), float(num)

  fraction = 1 / (num - idx + 1)
  print(" [*] GPU : %.4f" % fraction)
  return fraction

gpu_options = tf.GPUOptions(
per_process_gpu_memory_fraction=calc_gpu_fraction(FLAGS.gpu_fraction))
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
up_lanes = [3.5/2,3.5/2 + 3.5,250+3.5/2, 250+3.5+3.5/2, 500+3.5/2, 500+3.5+3.5/2]
down_lanes = [250-3.5-3.5/2,250-3.5/2,500-3.5-3.5/2,500-3.5/2,750-3.5-3.5/2,750-3.5/2]
left_lanes = [3.5/2,3.5/2 + 3.5,433+3.5/2, 433+3.5+3.5/2, 866+3.5/2, 866+3.5+3.5/2]
right_lanes = [433-3.5-3.5/2,433-3.5/2,866-3.5-3.5/2,866-3.5/2,1299-3.5-3.5/2,1299-3.5/2]
arrayOfVeh = [20] # for train
  
width = 750
height = 1299

  
Env = Environ(down_lanes,up_lanes,left_lanes,right_lanes, width, height, arrayOfVeh[0])
Env.new_random_game(20)
action_all_with_power = np.zeros([20, 3, 2],dtype = 'int32')
action_all_with_power_training = np.zeros([20, 3, 2],dtype = 'int32')  
vehicleNumber = 0
logs = []

v2iChannel_Range = range(0,20)
v2vInterference_Range = range(20,40)
v2vChannel_Range = range(40,60)
NeighSelect_Range = range(60,80)
time_remaining_Range = range(80,81) 
load_remaining_Range = range(81,82) 
action_RB_Range = range(82,83)
action_Power_Range = range(83,84)
reward_train_Range = range(84,85)
v2i_RB_state = 20
v2v_RB_Interference_state  = 20
v2v_RB_state = 20
Neighborhood_Selected_RB = 20
remianTime = 1
load_remin_Time = 1

while (1):
    print('0 : Action [resource block(0~19), power(5, 10, 23)]')
    print('1 : Save Log, ')
    print('2 : loop Action [resource block(0~19), power(5, 10, 23), iters]')
    print('3 : loop Action Random [iters]')
    print('4 : loop Action DDQN [iters]')
    print('9 : Reset envisionment')
    
    control = input()
    destVehicle = 0
    if control[0] == '0':
        
        data = control.split(' ')
        resourceblock = float(data[1])
        power = float(data[2])

        state = Env.get_state([vehicleNumber,destVehicle], True, action_all_with_power_training, action_all_with_power) 
        print('***********Now state***********')
        print(f'{vehicleNumber} -> {Env.vehicles[vehicleNumber].destinations[destVehicle]}')
        print(f'selected : v2i channel {state[int(resourceblock)]}, v2v interfenrece {state[20 + int(resourceblock)]}, v2v chnnel {state[40 + int(resourceblock)]}')
        print('*******************************')

        action_all_with_power_training[vehicleNumber, 0 : 3, 0] = resourceblock
        action_all_with_power_training[vehicleNumber, 0 : 3, 1] = power 
        reward_train, reward_best, v2v_rewards_list = Env.act_for_training(action_all_with_power_training, [vehicleNumber, destVehicle])
        print('***********after state***********')
        nextState = Env.get_state([vehicleNumber,destVehicle], True, action_all_with_power_training, action_all_with_power)
        print(f'selected next : v2i channel {nextState[int(resourceblock)]}, v2v interfenrece {nextState[20 + int(resourceblock)]}, v2v chnnel {nextState[40 + int(resourceblock)]}')
        print('*******************************')
        select_maxValue  = np.max(v2v_rewards_list)
        candidateList_selRB = np.where(v2v_rewards_list == select_maxValue)

        print('***********candidate action***********')
        print('candidate rb (20), power(5, 10, 23) : ', [(candidate_selRB_Power % 20, int(np.floor(candidate_selRB_Power/20)))  for candidate_selRB_Power in candidateList_selRB[0]])
        print('**************************************')
        vehicleNumber += 1

        if vehicleNumber == 20:
            vehicleNumber = 0

        log = np.concatenate((state, (resourceblock, power)))
        logs.append(log)

    elif control[0] == '1':    
        
        raw_data = {}  

        for index in range(v2i_RB_state):
             raw_data[f'v2i RB state {index}'] = []

        for index in range(v2v_RB_Interference_state):
             raw_data[f'v2v RB Interference_state {index}'] = []

        for index in range(v2v_RB_state):
             raw_data[f'v2v RB state {index}'] = []

        for index in range(Neighborhood_Selected_RB):
             raw_data[f'Neighborhood Selected_RB {index}'] = []

        raw_data['remianTime'] = []
        raw_data['load_remin_Time'] = []

        raw_data['Selected RB'] = []
        raw_data['Selected Power (5, 10, 23)'] = []
        raw_data['reward train'] = []

        for logindex, log in enumerate(logs):
            for v2iCIndex, v2iChannelvalue in enumerate(log[v2iChannel_Range]):
                raw_data[f'v2i RB state {v2iCIndex}'].append(v2iChannelvalue)

            for v2vCIndex, v2vChannelvalue in enumerate(log[v2vInterference_Range]):
                raw_data[f'v2v RB Interference_state {v2vCIndex}'].append(v2vChannelvalue)

            for v2vCIndex, v2vChannelvalue in enumerate(log[v2vChannel_Range]):
                raw_data[f'v2v RB state {v2vCIndex}'].append(v2vChannelvalue)
            
            for RBCIndex, neighSel_count in enumerate(log[NeighSelect_Range]):
                raw_data[f'Neighborhood Selected_RB {RBCIndex}'].append(neighSel_count)

            for time_remainTime in log[time_remaining_Range]:
                raw_data['remianTime'].append(time_remainTime)

            for load_remainTime in log[load_remaining_Range]:
                raw_data['load_remin_Time'].append(load_remainTime)

            for Selected_RB in log[action_RB_Range]:
                raw_data['Selected RB'].append(Selected_RB)

            for Selected_Power in log[action_Power_Range]:
                raw_data['Selected Power (5, 10, 23)'].append(Selected_Power)

            for Reward in log[reward_train_Range]:
                raw_data['reward train'].append(Reward)

        pd_data = DataFrame(raw_data)
        pd_data.to_excel("state metrix.xlsx", sheet_name='Sheet1')

        logs.clear()

    elif control[0] == '2': 
        
        data = control.split(' ')
        resourceblock = float(data[1])
        power = float(data[2])
        iter = int(data[3])

        for _ in range(iter):
            
            for dstVehicle_index in range(3):
                state = Env.get_state([vehicleNumber,dstVehicle_index], True, action_all_with_power_training, action_all_with_power) 

                print('***********Now state***********')
                print(f'{vehicleNumber} -> {Env.vehicles[vehicleNumber].destinations[dstVehicle_index]}')
                print(f'selected : v2i channel {state[int(resourceblock)]}, v2v interfenrece {state[20 + int(resourceblock)]}, v2v chnnel {state[40 + int(resourceblock)]}')
                print('*******************************')

                action_all_with_power_training[vehicleNumber, 0 : dstVehicle_index, 0] = resourceblock
                action_all_with_power_training[vehicleNumber, 0 : dstVehicle_index, 1] = power

                reward_train, reward_best, v2v_rewards_list = Env.act_for_training(action_all_with_power_training, [vehicleNumber, dstVehicle_index])
                
                select_maxValue  = np.max(v2v_rewards_list)
                candidateList_selRB = np.where(v2v_rewards_list == select_maxValue)

                print('***********candidate action***********')
                print('candidate rb (20), power(5, 10, 23) : ', [(candidate_selRB_Power % 20, int(np.floor(candidate_selRB_Power/20)))  for candidate_selRB_Power in candidateList_selRB[0]])
                print('**************************************')
                log = np.concatenate((state, [resourceblock, power, reward_train]))
                logs.append(log)

            vehicleNumber += 1

            if vehicleNumber == 20:
                vehicleNumber = 0

    elif control[0] == '3': 
        
        data = control.split(' ')
        iter = int(data[1])

        for _ in range(iter):
            
            for dstVehicle_index in range(3):
                state = Env.get_state([vehicleNumber,dstVehicle_index], True, action_all_with_power_training, action_all_with_power) 
                resourceblock = random.randrange(0,20)
                power = random.randrange(0,3)
                print('***********Now state***********')
                print(f'{vehicleNumber} -> {Env.vehicles[vehicleNumber].destinations[dstVehicle_index]}')
                print(f'selected : v2i channel {state[int(resourceblock)]}, v2v interfenrece {state[20 + int(resourceblock)]}, v2v chnnel {state[40 + int(resourceblock)]}')
                print('*******************************')

                action_all_with_power_training[vehicleNumber, 0 : dstVehicle_index, 0] = resourceblock
                action_all_with_power_training[vehicleNumber, 0 : dstVehicle_index, 1] = power

                reward_train, reward_best, v2v_rewards_list = Env.act_for_training(action_all_with_power_training, [vehicleNumber, dstVehicle_index])
                
                select_maxValue  = np.max(v2v_rewards_list)
                candidateList_selRB = np.where(v2v_rewards_list == select_maxValue)

                print('***********candidate action***********')
                print('candidate rb (20), power(5, 10, 23) : ', [(candidate_selRB_Power % 20, int(np.floor(candidate_selRB_Power/20)))  for candidate_selRB_Power in candidateList_selRB[0]])
                print('**************************************')
                log = np.concatenate((state, [resourceblock, power, reward_train]))
                logs.append(log)

            vehicleNumber += 1

            if vehicleNumber == 20:
                vehicleNumber = 0

            

    elif control[0] =='4':

        data = control.split(' ')
        iter = int(data[1])
        

        with tf.Session(config=config) as sess:
            config = []
            agent = Agent(config, Env, sess)
            agent.training = False
            agent.load_weight_from_pkl()

            for _ in range(iter):
                for dstVehicle_index in range(3):
                    state = Env.get_state([vehicleNumber,dstVehicle_index], True, action_all_with_power_training, action_all_with_power) 
                    action = agent.predict(state, 0, test_ep=True)
                    resourceblock = action % 20
                    power = int(np.floor(action/20))  
                    print('***********Now state***********')
                    print(f'{vehicleNumber} -> {Env.vehicles[vehicleNumber].destinations[dstVehicle_index]}')
                    print(f'selected : v2i channel {state[int(resourceblock)]}, v2v interfenrece {state[20 + int(resourceblock)]}, v2v chnnel {state[40 + int(resourceblock)]}')
                    print('*******************************')

                    action_all_with_power_training[vehicleNumber, 0 : dstVehicle_index, 0] = resourceblock
                    action_all_with_power_training[vehicleNumber, 0 : dstVehicle_index, 1] = power

                    reward_train, reward_best, v2v_rewards_list = Env.act_for_training(action_all_with_power_training, [vehicleNumber, dstVehicle_index])
                    
                    select_maxValue  = np.max(v2v_rewards_list)
                    candidateList_selRB = np.where(v2v_rewards_list == select_maxValue)

                    print('***********candidate action***********')
                    print('candidate rb (20), power(5, 10, 23) : ', [(candidate_selRB_Power % 20, int(np.floor(candidate_selRB_Power/20)))  for candidate_selRB_Power in candidateList_selRB[0]])
                    print('**************************************')
                    log = np.concatenate((state, [resourceblock, power, reward_train]))
                    logs.append(log)
            
                vehicleNumber += 1

                if vehicleNumber == 20:
                    vehicleNumber = 0

    elif control == '9':
        Env.new_random_game(20)
        action_all_with_power = np.zeros([20, 3, 2],dtype = 'int32')
        action_all_with_power_training = np.zeros([20, 3, 2],dtype = 'int32')  
        vehicleNumber = 0
        print('************renewchannel!************')    