# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 12:08:31 2021

@author: Handonghee
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

#맵에 상태를 sofrmax 확률로 변환 시키는 함수
def softmax_convert_into_pi_from_theta(theta):
    beta = 1.0
    [m,n] = theta.shape
    pi = np.zeros((m,n))
    
    exp_theta =  np.exp(beta*theta)
    
    for i in range(0,m):
        pi[i,:] = exp_theta[i,:] / np.nansum(exp_theta[i,:])
        
    pi = np.nan_to_num(pi)
    
    return pi

#start 행동 판단 함수
def get_next_s(pi,s):
    direction=["up","right","down","left"]
    
    next_direction = np.random.choice(direction, p=pi[s,:])
    
    if next_direction == "up":
        s_next = s-3
    elif next_direction == "right":
        s_next = s+1
    elif next_direction =="down":
        s_next = s+3
    elif next_direction =="left":
        s_next = s-1
    
    return s_next

def get_action_and_next_s(pi,s):
    direction=["up","right","down","left"]
    
    next_direction = np.random.choice(direction, p=pi[s,:])
    
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
    
    return [action, s_next]
#end 행동 판단 함수

#start 테스트 시뮬레이션 및 반복 함수
def goal_maze(pi):
    s = 0
    state_history = [0]
    
    while(1):
        next_s = get_next_s(pi,s)
        state_history.append(next_s)
        
        #######################

        
        #######################
        
        
        if next_s == 8:
            break
        else:
            s=next_s
            
    return state_history

def goal_maze_ret_s_a(pi):
    s = 0
    s_a_history = [[0,np.nan]]
   
    while(1):
        [action, next_s] = get_action_and_next_s(pi, s)
        s_a_history[-1][1] = action
        s_a_history .append([next_s,np.nan])
        
       
        
        if next_s == 8:
            break
        else:
            s=next_s
            
    return s_a_history
#end 테스트 시뮬레이션 및 반복 함수

#theta를 수정하는 함수
def update_theta(theta,pi,s_a_history):
    eta = 0.1 #학습률
    T = len(s_a_history) - 1 #목표 지점에 이르기까지 걸린 단계 수
    
    [m,n] = theta.shape
    delta_theta = theta.copy()
    
    for i in range(0,m):
        for j in range(0,n):
            if not(np.isnan(theta[i,j])):
                SA_i = [SA for SA in s_a_history if SA[0] == i]
                SA_ij = [SA for SA in s_a_history if SA == [i,j]]
                
                N_i = len(SA_i)
                N_ij = len(SA_ij)
                
                delta_theta[i,j] = (N_ij - pi[i,j] * N_i)/T
                
    new_theta = theta + eta*delta_theta
    
    return new_theta



stop_epsilon = 10**-4

pi_0_softmax = softmax_convert_into_pi_from_theta(theta_0)
theta = theta_0
pi = pi_0_softmax
is_continue = True
cout = 1

while is_continue:
    s_a_history = goal_maze_ret_s_a(pi)
    new_theta = update_theta(theta,pi,s_a_history)
    new_pi = softmax_convert_into_pi_from_theta(new_theta)
    
    print(np.sum(np.abs(new_pi-pi)))
    print("목표 지점 까지 움직인 횟수 : " + str(len(s_a_history) - 1) + "번")
    
    if np.sum(np.abs(new_pi - pi)) < stop_epsilon:
        is_continue = False
    else:
        theta=new_theta
        pi = new_pi

print("Test")
print(pi)

#pi_0_softmax = softmax_convert_into_pi_from_theta(theta_0)
#s_a_history = goal_maze_ret_s_a(pi_0_softmax)
#정책 수정
#new_theta = update_theta(theta_0, pi_0_softmax, s_a_history)
#pi = softmax_convert_into_pi_from_theta(new_theta)
#print(pi)


#goal_maze(pi_0)
#print(state_history)
#print("움직임 횟수 : " + str(len(state_history) - 1))