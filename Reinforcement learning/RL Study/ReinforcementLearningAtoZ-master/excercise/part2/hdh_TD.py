# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 21:04:04 2021

@author: Donghee
1 1Step TD, NStep TD를 분석 구현 하고
그래프를 그려본다
- TD Error의 값의 분산 그래프
- TD 구현 및 써보기
"""

import numpy as np
import matplotlib.pyplot as plt

from temporal_difference import TDAgent
from hdhenvs.gridworld import GridworldEnv
from utils.grid_visualization import visualize_value_function, visualize_policy

from tensorized_dp import TensorDP

def run_episodes(env, agent, total_eps, log_step):
    values = []
    log_iters = []
    
    for i in range(total_eps + 1):
        run_episode(env,agent)
        
        if i % log_step == 0:
            values.append(agent.v.copy())
            log_iters.append(i)
    info = dict()
    info['values'] = values
    info['iters'] = log_iters
    return info

def run_episode(env, agent):
    env.reset()
    
    while True:
        state = env.observe()
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        agent.sample_update(state, action, reward, next_state, done)

        if done:
            break
        
def run_nStepTD_episodes(env, agent, total_eps, log_step):
    values = []
    log_iters = []
    
    for i in range(total_eps + 1):
        run_nStepTD_episode(env,agent)
        
        if i % log_step == 0:
            values.append(agent.v.copy())
            log_iters.append(i)
            
    info = dict()
    info['values'] = values
    info['iters'] = log_iters
    return info
        
def run_nStepTD_episode(env, agent):
    """
    nStep TD의 실행함수
    MC 처럼 모든 state, action, reward를 episode로 모으고
    MC는 현재상태 이후 모든 G 값을 적용하여 업데이트 했던 방면
    nStep TD는 n 스탭만큼의 G값만 적용하여 업데이

    """
    env.reset()
    states = []
    actions = []
    rewards = []
    
    while True:
        state = env.observe()
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        
        if done:
            break
    episodes = (states, actions, rewards)
    agent.update(episodes)
                
        
np.random.seed(0)

nx, ny = 4,4
env = GridworldEnv([nx,ny])

td_agent = TDAgent(gamma=1.0, num_states=nx*ny, num_actions=4, epsilon=1.0, lr=1e-2, n_step=1)

"""
#1Step TD와 DP state value function 비교 하기
"""

total_eps = 1 # 10000
log_every = 1 # 1000

td_agent.reset_values()

dp_agent = TensorDP()
dp_agent.set_env(env)
v_pi = dp_agent.policy_evaluation()

info = run_episodes(env, td_agent, total_eps, log_every)

log_iters = info['iters']
mc_state_values=info['values']

n_rows = len(log_iters)
figsize_multiplier = 4

fig, ax = plt.subplots(n_rows, 2 , figsize=(2*figsize_multiplier, n_rows*figsize_multiplier))

#DP와 TD 비교
for viz_i, i in enumerate(log_iters):
    visualize_value_function(ax[viz_i,0], mc_state_values[viz_i], nx, ny, plot_cbar=(False))
    _ = ax[viz_i, 0].set_title("TD_PE after {} episodes".format(i), size=15)
    
    visualize_value_function(ax[viz_i,1], v_pi, nx, ny, plot_cbar=(False))
    _ = ax[viz_i, 1].set_title("DP_PE", size=15)


"""
#1Step TD 그래프 그리기
"""
reps = 10
values_over_run = []
total_eps = 3000 # 3000
log_every = 30 # 30


for i in range(reps):
    td_agent.reset_values()
    print("star to run {} th experiment ...".format(i))
    info = run_episodes(env, td_agent, total_eps, log_every)
    values_over_run.append(info['values'])
    
values_over_runs = np.stack(values_over_run)

v_pi_expanded = np.expand_dims(v_pi, axis=(0,1))

errors = np.linalg.norm(values_over_runs - v_pi_expanded, axis = -1)
error_mean = np.mean(errors, axis=0)
error_std = np.std(errors, axis= 0)

fig, ax = plt.subplots(1,1, figsize=(10,5))
ax.grid()
ax.fill_between(x=info['iters'],
                y1 = error_mean + error_std,
                y2 = error_mean - error_std,
                alpha=0.3)

ax.plot(info['iters'], error_mean, label="Evaluation error")
ax.legend()
_ = ax.set_xlabel('episodes')
_ = ax.set_ylabel('Errors')


"""
#NStep TD와 DP state value function 비교하기
"""
n_steps = 5
n_steps_td_agent = TDAgent(gamma=1.0, num_states=nx*ny, num_actions=4, epsilon=1.0, lr=1e-2, 
                           n_step=n_steps)

n_steps_td_agent.reset_values()
#2000, 500
info = run_episodes(env, n_steps_td_agent, 1, 1)

log_iters = info['iters']
mc_values=info['values']

n_rows = len(log_iters)
figsize_multiplier = 4

fig, ax = plt.subplots(n_rows, 2 , figsize=(2*figsize_multiplier, n_rows*figsize_multiplier))

for viz_i, i in enumerate(log_iters):
    visualize_value_function(ax[viz_i,0], mc_values[viz_i], nx, ny, plot_cbar=(False))
    _ = ax[viz_i, 0].set_title("nStep-TD_PE after {} episodes".format(i), size=15)
    
    visualize_value_function(ax[viz_i,1], v_pi, nx, ny, plot_cbar=(False))
    _ = ax[viz_i, 1].set_title("DP_PE", size=15)
    
"""
#1Step TD와 5Step TD의 10Step TD컨버전스 비교
"""
reps = 10
values_over_runs = []
total_eps = 3000
log_every = 30

for i in range(reps):
    n_steps_td_agent.reset_values()
    print("star to run {} th experiment ...".format(i))
    info = run_nStepTD_episodes(env, n_steps_td_agent, total_eps, log_every)
    values_over_runs.append(info['values'])
    
n_step_values_over_runs = np.stack(values_over_runs)

n_step_errors = np.linalg.norm(n_step_values_over_runs - v_pi_expanded, axis = -1)
n_step_error_mean = np.mean(n_step_errors, axis=0)
n_step_error_std = np.std(n_step_errors, axis= 0)

"""
10Step TD
"""

reps = 10
values_over_10_runs = []
total_eps = 3000
log_every = 30

n10_steps = 10
n_steps_td_agent = TDAgent(gamma=1.0, num_states=nx*ny, num_actions=4, epsilon=1.0, lr=1e-2, 
                           n_step=n10_steps)

n_steps_td_agent.reset_values()

#2000, 500
info = run_episodes(env, n_steps_td_agent, 1, 1)

log_iters = info['iters']
mc_values=info['values']

for i in range(reps):
    n_steps_td_agent.reset_values()
    print("star to run {} th experiment ...".format(i))
    info = run_nStepTD_episodes(env, n_steps_td_agent, total_eps, log_every)
    values_over_10_runs.append(info['values'])
    
n10_step_values_over_runs = np.stack(values_over_10_runs)

n10_step_errors = np.linalg.norm(n10_step_values_over_runs - v_pi_expanded, axis = -1)
n10_step_error_mean = np.mean(n10_step_errors, axis=0)
n10_step_error_std = np.std(n10_step_errors, axis= 0)


fig, ax = plt.subplots(1,1, figsize=(10,5))
ax.grid()

ax.fill_between(x=info['iters'],
                y1 = error_mean + error_std,
                y2 = error_mean - error_std,
                alpha=0.3)

ax.plot(info['iters'], error_mean, label="1-step TD Evaluation error")

ax.fill_between(x=info['iters'],
                y1 = n_step_error_mean + n_step_error_std,
                y2 = n_step_error_mean - n_step_error_std,
                alpha=0.3)

ax.plot(info['iters'], n_step_error_mean, label="{}-step TD Evaluation error".format(5))

ax.fill_between(x=info['iters'],
                y1 = n10_step_error_mean + n10_step_error_std,
                y2 = n10_step_error_mean - n10_step_error_std,
                alpha=0.3)

ax.plot(info['iters'], n10_step_error_mean, label="{}-step TD Evaluation error".format(10))


ax.legend()
_ = ax.set_xlabel('episodes')
_ = ax.set_ylabel('Errors with DP')
