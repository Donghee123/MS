# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 08:50:33 2021

@author: Hyewon Lee
"""

import sys; sys.path.append('..')

import gym
import torch
import matplotlib.pyplot as plt

from MLP import MultiLayerPerceptron as MLP
from common.train_utils import to_tensor
from common.memory.memory import ReplayMemory
from DQN import DQN

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

lr = 1e-4 * 5
batch_size = 256
gamma = 1.0
memory_size = 50000
total_eps = 3000
eps_max = 0.08
eps_min = 0.01

sampling_only_util = 2000

#10번 업데이트 할때마다 target q-net에 업데이트
#값이 1이면 target update를 안쓰는거나 마찬가지임
target_update_interval = 10 

qnet = MLP(4,2, num_neurons=[128])
qnet_target = MLP(4, 2, num_neurons=[128])

qnet_target.load_state_dict(qnet.state_dict())
agent = DQN(4, 2, qnet = qnet, qnet_target = qnet_target, lr=lr, gamma=gamma, epsilon=1.0)
env = gym.make('CartPole-v1')
memory = ReplayMemory(memory_size)

print_every = 100

cumulative_reward = []
for n_epi in range(total_eps):
    epsilon = max(eps_min, eps_max - eps_min * (n_epi / 200))
    agent.epsilon = torch.tensor(epsilon)
    s = env.reset()
    cum_r = 0
    
    while True:
        s = to_tensor(s, size=(1,4))
        a = agent.get_action(s)
        ns, r, done, info = env.step(a)
        
        #경험 생성
        experience = (s, 
                      torch.tensor(a).view(1,1),
                      torch.tensor(r / 100.0).view(1,1),#r값을 스케일링함
                      torch.tensor(ns).view(1,4),
                      torch.tensor(done).view(1,1))
        
        #경험 저장
        memory.push(experience)
        
        s = ns
        cum_r += r
        if done:
            break
    
    cumulative_reward.append(cum_r)
    #메모리의 값이 충분히 찼다면 업데이트를 시킴
    if len(memory) >= sampling_only_util:
        sampled_exps = memory.sample(batch_size)
        sampled_exps = prepare_training_inputs(sampled_exps)
        agent.update(*sampled_exps)
    
    #qnet의 파라미터를 하이퍼 파라미터 수만큼 업데이트 했다면 target qnet에 qnet의 파라미터를 업데이트함
    if n_epi % target_update_interval == 0:
        qnet_target.load_state_dict(qnet.state_dict())
        
    if n_epi % print_every == 0:
        msg = (n_epi, cum_r, epsilon)
        print("Episode : {:4.0f} | Cumulative Reward : {:4.0f} | Epsilon : {:.3f}".format(*msg))

plt.plot(cumulative_reward)