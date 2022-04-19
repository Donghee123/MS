from __future__ import division, print_function
import random
#import tensorflow.compat.v1 as tf
import tensorflow as tf
from agent import Agent
from Environment import *
import pandas as pd
import csv
import os

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
    
if FLAGS.gpu_fraction == '':
  raise ValueError("--gpu_fraction should be defined")

def calc_gpu_fraction(fraction_string):
  idx, num = fraction_string.split('/')
  idx, num = float(idx), float(num)

  fraction = 1 / (num - idx + 1)
  print(" [*] GPU : %.4f" % fraction)
  return fraction

def main(_):

  up_lanes = [3.5/2,3.5/2 + 3.5,250+3.5/2, 250+3.5+3.5/2, 500+3.5/2, 500+3.5+3.5/2]
  down_lanes = [250-3.5-3.5/2,250-3.5/2,500-3.5-3.5/2,500-3.5/2,750-3.5-3.5/2,750-3.5/2]
  left_lanes = [3.5/2,3.5/2 + 3.5,433+3.5/2, 433+3.5+3.5/2, 866+3.5/2, 866+3.5+3.5/2]
  right_lanes = [433-3.5-3.5/2,433-3.5/2,866-3.5-3.5/2,866-3.5/2,1299-3.5-3.5/2,1299-3.5/2]
  
  width = 750
  height = 1299
  
  arrayOfVeh = [20]
  

  
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
        
        #학습 전
        v2i_Sumrate, v2v_Sumrate, probability = agent.play(n_step = 100, n_episode = 20, random_choice = False)
        
        sumrateV2IList.append(v2i_Sumrate)
        sumrateV2VList.append(v2v_Sumrate)
        
        probabilityOfSatisfiedV2VList.append(probability)
        
        #학습
        #agent.train()
        
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
    tf.app.run()
