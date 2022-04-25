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

#지수 이동 평균법으로 모델의 성능변화를 추적하기위한 Class
class EMAMeter:
    def __init__(self, alpha:float=0.5):
        self.s = None
        self.alpha = alpha
        
    def update(self,y):
        if self.s is None:
            self.s = y
        else:
            self.s = self.alpha * y + (1-self.alpha) * self.s

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_eps = 10000
print_every = 500
ema_factor = 0.5
ema = EMAMeter(ema_factor)

for ep in range(n_eps):
    env.reset()  # restart environment 환경 리셋
    cum_r = 0 # cumulative reward 누적 보상값
    while True:
        s = env.state
        s = torch.tensor(s).float().view(1, 4) # convert to torch.tensor 배치사이즈 : 1, 상태 사이즈 : 4
        a = agent.get_action(s)
        ns, r, done, info = env.step(a)

        ns = torch.tensor(ns).float()  # convert to torch.tensor
        agent.update_sample(s, a, r, ns, done)
        cum_r += r
        
        if done:
            ema.update(cum_r)

            if ep % print_every == 0:
                print("Episode {} || EMA: {} || EPS : {}".format(ep, ema.s, agent.epsilon))

            if ep >= 150:
                agent.epsilon *= 0.999
            break
env.close()

#학습한 모델 저장하기
SAVE_PATH = './naive_dqn.pt'
torch.save(agent.state_dict(), SAVE_PATH)

#모델 불러오기
qnet2 = MLP(input_dim=s_dim, output_dim= a_dim, num_neurons=[128],hidden_act="ReLU", out_act="Identity")
agent2 = NaiveDQN(state_dim=s_dim, action_dim=a_dim, qnet=qnet2, lr=1e-4,gamma=1.0,epsilon=1.0)

#decaying하는 입실론값과 파라미터값을 모두 가져옴
agent2.load_state_dict(torch.load(SAVE_PATH))
