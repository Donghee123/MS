# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 09:30:41 2021

@author: DH

MC 키포인트
state, action, reward에 대해서 episode 형식으로 튜플로 변환후 
sum_r을 state_value(state), action_value(state, action)를 저장하여 값을 업데이트 하는 것이 키포인트

"""
import numpy as np
import matplotlib.pyplot as plt

from monte_carlo import ExactMCAgent, MCAgent
from hdhenvs.gridworld import GridworldEnv
from utils.grid_visualization import visualize_value_function, visualize_policy

np.random.seed(0)

def run_episode(env, agent, timeout=1000):
    '''
    에피소드 1개 수행
    state, action, reward를 모두 저장 시킴 -> history 저장 기능
    episode라는 튜플로 state, action, reward를 묶음
    agent update episode를 이용해서
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
env = GridworldEnv([ny,nx])

#ExactMCAgent : vanilla 버전 MC
mc_agent = ExactMCAgent(gamma=1.0, num_states=nx * ny, num_actions=4, epsilon = 1.0)

#actions : 행동 들
action_mapper = {
    0 : 'UP',
    1 : 'RIGHT',
    2 : 'DOWN',
    3 : 'LEFT'
}

''' 
거이 표준화 됨
에이전트 환경 interaction (MC 구조)
{
반복 : 
    에피소드 시작 -> env.reset() 환경 초기화
    반복 : 
        현재 상태 <= 환경으로 부터 현재 상태 관측 -> env.observe() env의 현재 state 리턴
        현재 행동 <= 에이전트의 정책함수 (현재 상태) -> 특정 Agnet의 GetAction(현재 state) 함수들을 사용, 내가 코딩한 Agnet class를 주로 이용
        다음 상태, 보상 <= 환경에 '형재 행동'을 가함 -> env.step(action) env에게 action을 가함
        if 다음 상태 == 종결 상태
            반복문 탈출
    
    에이전트의 가치함수 평가 및 정책함수 개선
}       
'''
env.reset()
step_counter = 0

"""
while True:
    
    print("At t = {}".format(step_counter))
    env._render()
    
    cur_state = env.observe()
    action = mc_agent.get_action(cur_state)
    next_state, reward, done, info = env.step(action)
    
    print('state : {}'.format(cur_state))
    print('action : {}'.format(action_mapper[action]))
    print('reward : {}'.format(reward))
    print('next state : {}\n'.format(next_state))
    step_counter += 1
    
    if done:
        break
"""

#바닐라 MC 버전
mc_agent.reset_statistics()

#5000번의 에피소드 
for _ in range(5000):
    #1번의 에피소드 (random state ~ Terminal state)
    run_episode(env, mc_agent)
    
mc_agent.compute_values()

print(mc_agent.v)


fig, ax = plt.subplots(1,2, figsize=(12,6))
visualize_value_function(ax[0], mc_agent.v, nx, ny)
_ = ax[0].set_title("Monte-carlo Policy evalutation")

#increamental MC 버전 
incre_mc_agent = MCAgent(gamma=1.0, lr=1e-3, num_states = nx * ny, num_actions=4,epsilon=1.0)
incre_mc_agent.reset_statistics()

#5000번의 에피소드 
for _ in range(5000):
    #1번의 에피소드 (random state ~ Terminal state)
    run_episode(env, incre_mc_agent)
    

print(incre_mc_agent.v)

visualize_value_function(ax[1], incre_mc_agent.v, nx, ny)
_ = ax[1].set_title("Increamental Monte-carlo Policy evalutation")