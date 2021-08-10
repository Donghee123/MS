# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 10:52:42 2021

@author: Hyewon Lee
SARSA : 
    On-policy
    현재 평가하는 정책과 행동 정책이 같음
Q-Learning : 
    Off-policy
    현재 평가하는 정책과 행동 정책이 다름 
    평가하는 정책(exploitation policy)은 그리디, 행동 정책(exploration policy)은 입실론 그리디를 이용
    
"""

import numpy as np
import matplotlib.pyplot as plt


from temporal_difference import SARSA, QLearner
from gym.envs.toy_text import CliffWalkingEnv
from utils.grid_visualization import visualize_policy, visualize_value_function

np.random.seed(0)

def run_sarsa(agent, env):
    env.reset()
    reward_sum = 0
    while True:
        state = env.s
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        next_action = sarsa_agent.get_action(next_state)
        reward_sum += reward
        
        agent.update_sample(state = state, action = action, reward = reward, next_state = next_state, next_action = next_action, done = done)
        
        if done:
            break
    
    return reward_sum

def run_qlearning(agent, env):
    env.reset()
    reward_sum = 0
    
    while True:
        state = env.s
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        reward_sum += reward
        
        agent.update_sample(state = state, action = action, reward = reward, next_state = next_state, done = done)
        
        if done:
            break
    
    return reward_sum


cliff_env = CliffWalkingEnv()

#cliff 환경을 보여줌
cliff_env.render()
"""
o  o  o  o  o  o  o  o  o  o  o  o            o : 이동 가능
o  o  o  o  o  o  o  o  o  o  o  o            x : 시작 지점
o  o  o  o  o  o  o  o  o  o  o  o            c : 절벽, 해당지점으로 이동시 o로 이동
x  C  C  C  C  C  C  C  C  C  C  T            T : Terminal state
"""

sarsa_agent = SARSA(gamma=0.9, num_states=cliff_env.nS, num_actions = cliff_env.nA, epsilon=0.1, lr=1e-1)
q_agent = QLearner(gamma=0.9, num_states=cliff_env.nS, num_actions = cliff_env.nA, epsilon=0.1, lr=1e-1)

num_eps = 1500

sarsa_rewards = []
qlearning_rewards = []

for i in range(num_eps):
    sarsa_reward_sum = run_sarsa(sarsa_agent, cliff_env)
    qlearning_reward_sum = run_qlearning(q_agent, cliff_env)
    
    sarsa_rewards.append(sarsa_reward_sum)
    qlearning_rewards.append(qlearning_reward_sum)
    
fig, ax = plt.subplots(1,1, figsize=(10,5))
ax.grid()
ax.plot(sarsa_rewards, label = 'SARSA  episode reward')
ax.plot(qlearning_rewards, label = 'Q-Learning episode reward', alpha = 0.5)
ax.legend()

fig, ax = plt.subplots(2,1, figsize=(20,10))
visualize_policy(ax[0], sarsa_agent.q, cliff_env.shape[0], cliff_env.shape[1])
_ = ax[0].set_title("SARSA policy")
visualize_policy(ax[1], q_agent.q, cliff_env.shape[0], cliff_env.shape[1])
_ = ax[1].set_title("Q-Learning greedy policy")
