# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 12:08:31 2021

@author: Handonghee
@subject : miro epsilon greedy search based on "SARSA" 
           1. epsilon greedy
           2. Sarsa
             - eta : stepsize
             - r : reward
             - Q[s,a] : action value function
             - gamma : discount facetor
             - s, a : current sate, action
             - s_next,a_next : next sate, action
             - Q[s,a] = Q[s,a] + eta * (r + gamma * Q[s_next,a_next] - Q[s,a])
             
"""

import numpy as np
import matplotlib.pyplot as plt
import time

fig = plt.figure(figsize=(5,5))
ax = plt.gca()

plt.plot([1,1],[0,1], color='red', linewidth=2)
plt.plot([1,2],[2,2], color='red', linewidth=2)
plt.plot([2,2],[2,1], color='red', linewidth=2)
plt.plot([2,3],[1,1], color='red', linewidth=2)

plt.text(0.5,2.5 ,'S0',size=14, ha='center')
plt.text(1.5,2.5,'S1',size=14, ha='center')
plt.text(2.5,2.5,'S2',size=14, ha='center')
plt.text(0.5,1.5,'S3',size=14, ha='center')
plt.text(1.5,1.5,'S4',size=14, ha='center')
plt.text(2.5,1.5,'S5',size=14, ha='center')
plt.text(0.5,0.5,'S6',size=14, ha='center')
plt.text(1.5,0.5,'S7',size=14, ha='center')
plt.text(2.5,0.5,'S8',size=14, ha='center')
plt.text(0.5,2.3,'START',size=14, ha='center')
plt.text(2.5,0.3,'GOAL',size=14, ha='center')


ax.set_xlim(0,3)
ax.set_ylim(0,3)
plt.tick_params(axis='both', which='both', bottom=False, top=False,
                labelbottom=False, right=False, left=False, labelleft=False)

line, = ax.plot([0.5],[2.5],marker='o', color='g', markersize=60)

#각 좌표의 이동가능한 상태를 의미 (상, 우, 하, 좌)(시계방향)
theta_0 = np.array([[np.nan, 1, 1, np.nan],    #s0
                   [np.nan, 1, np.nan, 1],     #s1
                   [np.nan, np.nan, 1, 1],     #s2
                   [1, 1, 1, np.nan],          #s3
                   [np.nan, np.nan, 1, 1],     #s4
                   [1, np.nan, np.nan, np.nan],#s5
                   [1, np.nan, np.nan, np.nan],#s6
                   [1, 1, np.nan, np.nan],])   #s7


#맵에 상태를 랜덤 확률로 변환 시키는 함수
def simple_convert_into_pi_from_theta(theta):
    [m,n] = theta.shape
    pi = np.zeros((m,n))
    for i in range(0,m):
        pi[i,:] = theta[i,:]/np.nansum(theta[i,:])
        
    pi = np.nan_to_num(pi)
    
    return pi

#start 행동 판단 함수
def get_s_next(s,a, Q,epsilon,pi_0):
    direction=["up","right","down","left"]
    
    next_direction = direction[a]
    
    if next_direction == "up":
        s_next = s-3
    elif next_direction == "right":
        s_next = s+1
    elif next_direction =="down":
        s_next = s+3
    elif next_direction =="left":
        s_next = s-1
    
    return s_next

#choice action based on epsilon - greedy  
def get_action(s, Q, epsilon, pi_0):
    direction=["up","right","down","left"]
    
    #epsilong-greedy 기반 행동 결정 
    if np.random.rand() < epsilon:
        #무작위 선택
        next_direction = np.random.choice(direction, p = pi_0[s,:])
    else:
        #Action value function 값이 최대가 되는 행동을 선택
        next_direction = direction[np.nanargmax(Q[s,:])]
    
    if next_direction == "up":
        action = 0
        s_next = s-3
    elif next_direction == "right":
        action = 1
        s_next = s+1
    elif next_direction =="down":
        action = 2
        s_next = s+3
    elif next_direction =="left":
        action = 3
        s_next = s-1
    
    return action
#end 행동 판단 함수

#start 테스트 시뮬레이션 및 반복 함수
def goal_maze(pi):
    s = 0
    state_history = [0]
    
    while(1):
        next_s = get_s_next(pi,s)
        state_history.append(next_s)
        
        #######################

        
        #######################
        
        
        if next_s == 8:
            break
        else:
            s=next_s
            
    return state_history

def getReward(next_s):
    if next_s == 8:
        return 1
    else:
        return 0
    
def goal_maze_ret_s_a_Q(Q, epsilon, eta, gamma, pi):
    """
    한번의 에피소드 수행
    터미널 스테이트에 도달하면 종료
    """
    # 시작 위치
    s = 0 
    a = a_next = get_action(s, Q, epsilon, pi)
    s_a_history = [[s,np.nan]]
   
    while(1):
        
        #다음 행동 저장
        a = a_next
        #가장 마지막 히스토리에 다음 행동 저장
        s_a_history[-1][1] = a
        
        #현재 상태 s에서 선택한 a에 따라 다음 상태로 변하는 것
        s_next = get_s_next(s,a,Q,epsilon,pi)
        
        #다음 상태 히스토리에 추가, 액션은 아직 미설정해서 냅둠
        s_a_history.append([s_next,np.nan])
        
        #보상 계산
        reward = getReward(s_next)
        
        #만약 다음 상태가 terminal state 면 다음 ation에 nan을 줌
        if reward == 1:
            a_next = np.nan
        else:
            #다음 action 계산
            a_next = get_action(s_next,Q,epsilon,pi)

        #Sarsa 업데이트
        Q = Sarsa(s,a,reward,s_next,a_next,Q,eta,gamma)
        
        if s_next == 8:
            break
        else:
            s=s_next
            
    return [s_a_history,Q]

#end 테스트 시뮬레이션 및 반복 함수

def Sarsa(s, a, r, s_next, a_next, Q, eta,gamma):
    
    #다음 상태가 terminal state라면
    if s_next == 8:
        Q[s,a] = Q[s,a] + eta * (r-Q[s,a])
    else:#다음 상태가 termianl state가 아니면 다음 상태에서 선정한 액션으로 업데이트
        Q[s,a] = Q[s,a] + eta * (r + gamma * Q[s_next,a_next] - Q[s,a])
    return Q



stop_epsilon = 10**-4

#theta의 값을 단순 확률 형식으로 변환하여 저장하는 policy
pi_0 = simple_convert_into_pi_from_theta(theta_0)

#Action value function 초기화
[a,b] = theta_0.shape
Q = np.random.rand(a,b) * theta_0

eta = 0.1
gamma = 0.9
epsilon = 0.5
v = np.nanmax(Q,axis = 1) #각 상태마다 가치의 최대값을 계산
is_continue = True
episode = 1

while is_continue:
    print("에피소드: " + str(episode))
    
    #Sarsa 알고리즘 수행 1번의 episode
    [s_a_history, Q] = goal_maze_ret_s_a_Q(Q, epsilon, eta, gamma, pi_0)
    
    #Action value function의 변화 보기
    new_v = np.nanmax(Q,axis = 1)
    
    #Action value function의 변화량 보기
    print(np.sum(np.abs(new_v - v)))
    v = new_v
    
    print("목표 지점에 이르기까지 걸린 단계수는 " + str(len(s_a_history) - 1) + " 단계 입니다.")
    
    #100 에피소드 반복
    episode = episode + 1
    
    #100번 반복시 탈출
    if episode > 100:
        break
    
    #입실론값 낮춤
    epsilon = epsilon / 2
    
#초기 policy값 저장
