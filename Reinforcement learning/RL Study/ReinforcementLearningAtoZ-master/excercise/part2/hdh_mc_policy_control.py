# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 08:19:41 2021

@author: Donghee
1 MC Control
  - GLIE Epsilon Greedy
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from monte_carlo import ExactMCAgent, MCAgent
from hdhenvs.gridworld import GridworldEnv
from utils.grid_visualization import visualize_policy, visualize_value_function

np.random.seed(0)

def decaying_epsilon_and_run(agent, env, decaying_factor:float, n_runs:int = 5000):
    agent.decaying_epsilon(decaying_factor)
    agent.reset_statistics()
    
    print('epsilon : {}'.format(agent.epsilon))
    
    for _ in range(n_runs):
        run_episode(env, agent)
    
    agent.improve_policy()
    
    fix, ax = plt.subplots(1,2, figsize=(12,6))
    visualize_value_function(ax[0], agent.v, nx, ny)
    _ = ax[0].set_title("Value pi")
    visualize_policy(ax[1], agent.q, nx, ny)
    _ = ax[1].set_title("Greedy policy")
    

def highlight_max(s):
    is_max = s == s.max()
    return ['background-color: green' if v else '' for v in is_max]

def visualize_q(q):
    df = pd.DataFrame(q, columns=['up','right','down', 'left']).T
    df = df.style.apply(highlight_max)
    return df

def run_episode(env, agent, timeout=1000):
    '''
    에피소드 1개 수행
    state, action, reward를 모두 저장 시킴 -> history 저장 기능
    episode라는 튜플로 state, action, reward를 묶음
    agent update episode를 이용해서
    
    timeout은 무한정으로 수행하는 episode를 방어 하기 위함
    '''
    env.reset()
    states = []
    actions = []
    rewards = []

    i = 0
    timeouted = False
    while True:
        
        state = env.s
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)

        if done:
            break
        else:
            i += 1
            if i >= timeout:
                timeouted = True
                break

    if not timeouted:
        episode = (states, actions, rewards)
        agent.update(episode)
        
nx, ny = 4, 4
env = GridworldEnv([nx,ny])

mc_agent = MCAgent(gamma=1.0, num_states = nx * ny, num_actions=4, epsilon=1.0, lr=1e-3)

#5000 -> 0
for _ in range(5000):
    run_episode(env, mc_agent)
    
mc_agent.improve_policy()
mc_agent._policy_q

fix, ax = plt.subplots(1,2, figsize=(12,6))
visualize_value_function(ax[0], mc_agent.v, nx, ny)
_ = ax[0].set_title("Non decaying epsilon Value pi")
visualize_policy(ax[1], mc_agent.q, nx, ny)
_ = ax[1].set_title("Non decaying epsilon Greedy policy")

df = visualize_q(mc_agent.q)


"""
Decaying epsilon mc
입실론 감소율을 잘 조절하면 MC도 DP와 근사한 state value function을 구할 수 있다.
"""

decaying_epsilon_and_run(mc_agent, env, 0.9)
decaying_epsilon_and_run(mc_agent, env, 0.9)
decaying_epsilon_and_run(mc_agent, env, 0.1)
decaying_epsilon_and_run(mc_agent, env, 0.1)
decaying_epsilon_and_run(mc_agent, env, 0.1)
decaying_epsilon_and_run(mc_agent, env, 0.0)


"""
Decaying epsilon mc
입실론 감소율 0, 100000수행
"""
decaying_epsilon_and_run(mc_agent, env, 0.0,100000)

"""
성급한 입실론 감소 실험
epsilon greedy시 epsilon의 값을 exploration, exploitation의 비율을 잘 조절 하기위해
적절히 값을 지정해야한다.
"""
greedy_mc_agent = MCAgent(gamma=1.0, num_states=nx*ny, num_actions=4, epsilon=0.0, lr=1e-3)
decaying_epsilon_and_run(greedy_mc_agent, env, 0.0,5000)
