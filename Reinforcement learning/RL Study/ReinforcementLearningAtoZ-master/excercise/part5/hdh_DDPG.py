# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 09:01:41 2021

@author: Hyewon Lee
"""

import sys; sys.path.append('..')

import gym
import torch
import numpy as np
import matplotlib.pyplot as plt

from MLP import MultiLayerPerceptron as MLP


from DQN import prepare_training_inputs
from DDPG import DDPG, Actor, Critic
from DDPG import OrnsteinUhlenbeckProcess as OUProcess


from common.target_update import soft_update
from common.train_utils import to_tensor
from common.memory.memory import ReplayMemory

from IPython.display import HTML
HTML('<img src=>"images/pendulum.gif">')

"""
Pendulum
state : cos theta, sin theta, 각속도
action : -2.0 ~ +2.0
reward : state의 cos theta, sin theta, action 값이 0에 가까워 질수록 높은 보상을 받음.
"""


#저장 되어있는 파라미터를 불러올거면 False, 새로 학습 시킬거면 True
FROM_SCRATCH = False

#DDPG 학습은 시간이 매우 오래 걸림
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

lr_actor = 0.005
lr_critic = 0.001
gamma = 0.99
batch_size = 256
memory_size = 50000
tau = 0.001 # 소프트 업데이트의 타우
sampling_only_until = 2000

actor = Actor()
actor_target = Actor()
critic = Critic()
critic_target = Critic()

agent = DDPG(critic=critic,critic_target=critic_target,actor=actor, actor_target=actor_target,
             lr_actor=lr_actor, lr_critic=lr_critic, gamma = gamma)

memory = ReplayMemory(memory_size)

total_eps = 200
print_every = 10

env = gym.make('Pendulum-v0')

if FROM_SCRATCH:
    for n_epi in range(total_eps):
        ou_noise = OUProcess(mu=np.zeros(1))
        s = env.reset()
        cum_r = 0
        
        while True:
            s = to_tensor(s, size=(1,3)).to(DEVICE)
            a = agent.get_action(s).cpu().numpy() + ou_noise()[0]
            ns, r, done, info = env.step(a)
            
            experience = (s,
                          torch.tensor(a).view(1,1),
                          torch.tensor(r).view(1,1),
                          torch.tensor(ns).view(1,3),
                          torch.tensor(done).view(1,1),)
            
            memory.push(experience)
            
            s = ns
            cum_r += r
            
            if len(memory) >= sampling_only_until:
                #train agent
                sampled_exps = memory.sample(batch_size)
                sampled_exps = prepare_training_inputs(sampled_exps, device = DEVICE)
                agent.update(*sampled_exps)
                
                #update target networks
                #OUProcess에 의해 noisy 한 환경이 생김.
                #soft_update를 통해 noisy 한 환경에서 tau값을 조절하며 업데이트함
                soft_update(agent.actor, agent.actor_target, tau)
                soft_update(agent.critic, agent.critic_target, tau)
                
            if done:
                break
        if n_epi % print_every == 0:
            msg = (n_epi, cum_r) 
            print("Episode : {} | Cumulative REward : {} |".format(*msg))
            
    torch.save(agent.state_dict(), 'ddpg_cartpole_user_trained.ptb')
else:
    agent.load_state_dict(torch.load('ddpg_cartpole_user_trained.ptb'))
                          
env = gym.make('Pendulum-v0')

s = env.reset()
cum_r = 0

while True:
    s = to_tensor(s, size=(1, 3)).to(DEVICE)
    a = agent.get_action(s).to('cpu').numpy()
    ns, r, done, info = env.step(a)
    s = ns
    env.render()
    if done:
        break
    
env.close()