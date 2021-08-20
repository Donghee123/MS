# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 09:46:45 2021

@author: DH
"""

import sys; sys.path.append('..')


import gym
import numpy as np
import torch
import torch.nn as nn
#from IPython.display import YoutubeVideo

from MLP import MultiLayerPerceptron as MLP
from QLearner import NaiveDQN 

env = gym.make('CartPole-v1')
env.reset()

#random action수행
"""
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())
    
env.close()
"""

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.n

print('상태 공간 벡터: {}'.format(s_dim))
print('액션 공간 벡터: {}'.format(a_dim))

#DQN 트레이닝 수행
qnet = MLP(input_dim=s_dim, output_dim= a_dim, num_neurons=[128],hidden_act="ReLU", out_act="Identity")
#Agnet 생성
agent = NaiveDQN(state_dim=s_dim, action_dim=a_dim, qnet=qnet, lr=1e-4,gamma=1.0,epsilon=1.0)

