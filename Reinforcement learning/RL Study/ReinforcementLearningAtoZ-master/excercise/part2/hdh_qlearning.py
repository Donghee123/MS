# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 10:34:58 2021

@author: Hyewon Lee
"""

import numpy as np
import matplotlib.pyplot as plt


from temporal_difference import QLearner
from hdhenvs.gridworld import GridworldEnv
from utils.grid_visualization import visualize_policy, visualize_value_function

np.random.seed(0)

nx, ny = 4, 4
env = GridworldEnv([nx,ny])

qlearning_agent = QLearner(gamma=1.0, num_states=env.nS, num_actions = env.nA, epsilon =1.0, lr= 1e-1)

num_eps = 10000
report_every = 1000
qlearning_qs = []
iter_idx = []
qlearning_rewards = []

for i in range(num_eps):
    reward_sum = 0
    env.reset()
    while True:
        state = env.s
        action = qlearning_agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        
        qlearning_agent.update_sample(state=state, action=action, reward=reward, next_state=next_state, done=done)
        reward_sum += reward
        if done:
            break
    
    qlearning_rewards.append(reward_sum)
    
    if i % report_every == 0:
        print('Running {}th episode'.format(i))
        print('Reward sum : {}'.format(reward_sum))
        qlearning_qs.append(qlearning_agent.q.copy())
        iter_idx.append(i)
        
num_plots = len(qlearning_qs)
fig, ax = plt.subplots(num_plots, figsize=(num_plots*5*5, num_plots*5))
for i, (q, viz_i) in enumerate(zip(qlearning_qs, iter_idx)):
    visualize_policy(ax[i], q, env.shape[0], env.shape[1])
    _ = ax[i].set_title("Greedy policy at {} th episode".format(viz_i))