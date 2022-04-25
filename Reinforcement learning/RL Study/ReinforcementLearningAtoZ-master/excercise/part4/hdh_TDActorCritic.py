# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 12:40:20 2021

@author: DH
Vanilla version의 TD Actor-critic
REINFORCE 보다 매우 안좋음.
이유 : 
    1. 서로 다른 2개의 알고리즘을 비교하는 것이 문제임.
    2. REINFORCE에 비해서 하이퍼 파라미터의 튜닝이 더 필요함.
    3. Actor와 Critic 성능은 종속적이며 둘중 하나라도 안된다면 학습이 힘듬.
    다음 파트5에서 계선된것을 쓸것임.
"""

import sys; sys.path.append('..')

import gym
import torch
import matplotlib.pyplot as plt

from MLP import MultiLayerPerceptron as MLP
from ActorCritic import TDActorCritic
from common.train_utils import EMAMeter, to_tensor

env = gym.make('CartPole-v1')
s_dim = env.observation_space.shape[0]
a_dim = env.action_space.n

#입력 레이어 : 현재 상태의 값을 모두 받음
#출력 레이어 : 현재 상태에서의 행동 정책을 얻기 위함.
#히든 레이어 : 레이어 1개 (128노드)로 구성
policy_net = MLP(s_dim, a_dim, [128])

#입력 레이어 : 현재 상태의 값을 모두 받음
#출력 레이어 : 현재 상태의 가치 함수를 얻기 위함
#히든 레이어 : 레이어 1개 (128노드)로 구성
value_net = MLP(s_dim, 1, [128])

agent = TDActorCritic(policy_net, value_net)
ema = EMAMeter()

n_eps = 10000
print_every = 500

for ep in range(n_eps):
    s = env.reset()
    cum_r = 0
    
    while True:
        s = to_tensor(s,size=(1,4))
        a = agent.get_action(s)
        ns, r, done, info = env.step(a.item())
        
        ns = to_tensor(ns, size=(1,4))
        agent.update(s,a.view(-1,1),r, ns,done)
        
        s = ns.numpy()
        cum_r += r
        if done:
            break
    
    ema.update(cum_r)
    
    if ep % print_every == 0:
        print("Episode {} || EMA: {}".format(ep, ema.s))