# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 07:43:49 2021

@author: Hyewon Lee
"""

import sys; sys.path.append('..')

import gym
import torch
import matplotlib.pyplot as plt

from MLP import MultiLayerPerceptron as MLP
from PolicyGradient import REINFORCE
from common.train_utils import EMAMeter, to_tensor

env = gym.make('CartPole-v1')
s_dim = env.observation_space.shape[0]
a_dim = env.action_space.n

n_vars = 10
#평균이 0이고 표준편차가 1인 가우시안 분포를 생성
logits = torch.randn(n_vars)
max_idx = logits.argmax().item()

"""
Softmax 직접 구현 코드

#바닐라 Softmax

#softmax 계산 : 총합이 1인 결과값으로 변화
probs = torch.exp(logits) / torch.exp(logits).sum()
prob_max_idx = probs.argmax().item()

fig,ax = plt.subplots(4, 1)
ax[0].grid()
ax[0].plot(range(1, n_vars+1), logits)
ax[0].scatter(max_idx+1, logits[max_idx], c='orange')

ax[1].grid()
ax[1].plot(range(1, n_vars+1), probs)
ax[1].scatter(prob_max_idx+1, probs[prob_max_idx], c='orange')


#Temperature 파라미터 T를 적용한 softmax
큰 뉴럴 네트워크의 결과를 작은 뉴럴 네트워크와 연결시킬때 사용 한다함.
t1 = 10
#Temperature softmax 계산 : 총합이 1인 결과값으로 변화
probs = torch.exp(logits/t1) / torch.exp(logits/t1).sum()
t2 = 0.1
#Temperature softmax 계산 : 총합이 1인 결과값으로 변화
probs = torch.exp(logits/t2) / torch.exp(logits/t2).sum()

#exp의 단점은 입력값이 작거나/클때 쉽게 underflow/overflow의 염려가 있음.
"""

net = MLP(s_dim, a_dim, [128])
agent = REINFORCE(net)
ema = EMAMeter()

n_eps = 10000
print_every = 500

for ep in range(n_eps):
    s = env.reset()
    cum_r = 0
    
    states = []
    actions = []
    rewards = []
    
    #한번의 에피소드 수집
    while True:
        s = to_tensor(s, size=(1,4))
        a = agent.get_action(s)
        ns, r, done, info = env.step(a.item())
        
        states.append(s)
        actions.append(a)
        rewards.append(r)
        
        s = ns
        cum_r += r
        if done:
            break
        
    ema.update(cum_r)
    
    if ep % print_every == 0:
        print("Episode {} || EMA: {}".format(ep, ema.s))
        
    #업데이트를 위해 states, actions, rewards의 자료구조를 수정
    #REINFORCE는 1개의 에피소드에서 모든 샘플들을 사용해서 업데이트함.
    states = torch.cat(states, dim=0)
    actions = torch.stack(actions).squeeze()
    rewards = torch.tensor(rewards)
    
    #episode라는 튜플로 states, actions, rewards를 묶음
    episode = (states, actions, rewards)
    
    #업데이트!!
    #agent.update_episode(episode, use_norm=True)
    agent.update(episode)
        
