# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 17:28:28 2021

@author: DH
SARSA를 이용한
polocy control

"""

import numpy as np
import matplotlib.pyplot as plt

from temporal_difference import SARSA
from hdhenvs.gridworld import GridworldEnv
from utils.grid_visualization import visualize_value_function, visualize_policy

np.random.seed(0)

nx,ny = 4,4
env=GridworldEnv([nx,ny])

sarsa_agent = SARSA(gamma=1.0,lr=1e-1,num_states=env.nS,num_actions=env.nA,epsilon=1.0)

num_eps = 10000
report_every = 1000
sarsa_qs = []
iter_idx = []
sarsa_rewards = []

for i in range(num_eps):
    reward_sum = 0
    env.reset()
    while(True):
        state = env.s
        action = sarsa_agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        next_action = sarsa_agent.get_action(next_state)
        
        sarsa_agent.update_sample(state=state, action=action, reward=reward, next_state=next_state, next_action=next_action, done=done)
        reward_sum += reward
        if done:
            break
        
    sarsa_rewards.append(reward_sum)
    
    if i % report_every == 0:
        print('Running {} th episode'.format(i))
        print('Reward sum : {}'.format(reward_sum))
        sarsa_qs.append(sarsa_agent.q.copy())
        iter_idx.append(i)

num_plots = len(sarsa_qs)
fig, ax = plt.subplots(num_plots, figsize=(num_plots*5*5, num_plots*5))
for i, (q, viz_i) in enumerate(zip(sarsa_qs, iter_idx)):
    visualize_policy(ax[i], q, env.shape[0], env.shape[1])
    _ = ax[i].set_title("Greedy policy at {} th episode".format(viz_i))
    
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    