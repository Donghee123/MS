import pybullet_envs
import gym
import numpy as np
import torch as T
import torch.nn.functional as F
from torch.optim import Adam
from sac_torch import Agent as SAC
from utils import plot_learning_curve
from gym import wrappers
import argparse
import datetime
from gym import spaces
import numpy as np
import itertools
import torch
from Environment import *
#from torch.utils.tensorboard import SummaryWriter
import random
import tensorflow as tf
from dqnagent import Agent
import pandas as pd
import csv
import os

def calc_gpu_fraction(fraction_string):
  idx, num = fraction_string.split('/')
  idx, num = float(idx), float(num)

  fraction = 1 / (num - idx + 1)
  print(" [*] GPU : %.4f" % fraction)
  return fraction


flags = tf.app.flags

#======================DQN=====================================================
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
#======================DQN=====================================================

#======================SAC=====================================================
parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="HalfCheetah-v2",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                    help='Automaically adjust α (default: True)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000001)')
parser.add_argument('--hidden_size', type=int, default=512, metavar='N',
                    help='hidden size (default: 512)')
parser.add_argument('--replay_size', type=int, default=10000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
#테스트 관련 하이퍼파라미터==============================================================

#cuda 사용 유무
parser.add_argument('--cuda', action="store_true",default=True,
                    help='run on CUDA (default: False)')

#critic target update 주기
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N', # 1
                    help='Value target update per no. of updates per step (default: 1)')

#replay memory로 부터 데이터를 가져오는 갯수
parser.add_argument('--batch_size', type=int, default=256, metavar='N', # 256                    
                    help='batch size (default: 256)')

#학습 스탭당 actor, critic이 학습하는 횟수를 지정
parser.add_argument('--updates_per_step', type=int, default=15, metavar='N', 
                    help='model updates per simulator step (default: 1)')

#초기 랜덤 선택하는 수
parser.add_argument('--start_steps', type=int, default=1000, metavar='N',  # 10000
                    help='Steps sampling random actions (default: 10000)')

#총 학습 스탭, 1번의 스탭당 (차량 수(20) * 인접 차량 수(3) = 60)번 선택함 
parser.add_argument('--train_step', type=int, default=40000, metavar='N',  # 40000
                    help='Set train step (default: 40000)')

#학습 스탭에 따른 테스트 시뮬레이션 주기 
parser.add_argument('--test_step', type=int, default=100, metavar='N',  # 2000
                    help='Set test interval step (default: 2000)')

#학습 스탭에 따른 테스트 그래프 주기 
parser.add_argument('--train_graph_step', type=int, default=50, metavar='N',  # 2000
                    help='Set test interval step (default: 2000)')
#======================================================================================
#======================SAC=====================================================


args = parser.parse_args()

# Environment
# env = NormalizedActions(gym.make(args.env_name))

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
              right_lanes, width, height, nVeh)  # V2X 환경 생성


torch.manual_seed(args.seed)
np.random.seed(args.seed)
# End 초기화 부분

# SAC Agent state state 83 , DQN Agent state 82
statespaceSize = 83

action_space = spaces.Box(
    np.array([0.0]), np.array([23.0]), dtype=np.float32)

gpu_options = tf.GPUOptions(
per_process_gpu_memory_fraction=calc_gpu_fraction(FLAGS.gpu_fraction))
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    config = []
    dqnagent = Agent(config, env, sess)        
    sacagent = SAC(dqnagent, action_space, env, args)
    sacagent.train_with_dqn()
    


"""    
    if load_checkpoint:
        agent.load_models()
       

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn()
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)
"""        
    
