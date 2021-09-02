# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 14:25:25 2021

@author: Handonghee
"""
import numpy as np
from math import *
import matplotlib.pylab as plt

pBackup = dict()
def poisson(x, lam):
    global pBackup
    key = x * 10 + lam
    if key not in pBackup.keys():
        pBackup[key] = np.exp(-lam) * pow(lam, x) / factorial(x)
    return pBackup[key]

def expectedReturn(state, action, stateValue):
    """
    parameter
    state[첫번째 주차장 차량 수,두번째 주차장 차량 수]위치에서 action을 했을경우 푸아송 랜덤 프로세스 기반 Request, Return
    """
    # Initiate and populate returns with cost associated with moving cars
    returns = 0.0
    
    #이동하는 차량 갯수당 손실 보상 추가
    returns -= COST_OF_MOVING * np.absolute(action)
    
    #Number of cars to start the day
    #action 후 state[첫번째 주차장 차량 수,두번째 주차장 차량 수]
    carsLoc1 = int(min(state[0] - action, MAX_CARS))
    carsLoc2 = int(min(state[1] + action, MAX_CARS))
    
    # 차량 렌트 확률 반복
    for rentalsLoc1 in range(0, POISSON_UPPER_BOUND):
        for rentalsLoc2 in range(0, POISSON_UPPER_BOUND):
            
            # Rental Probabilities
            # 2사건(Request1,Request2)이 동시에 일어날 확률 이므로 Multiply
            rentalsProb = poisson(rentalsLoc1, EXPECTED_FIRST_LOC_REQUESTS) * poisson(rentalsLoc2, EXPECTED_SECOND_LOC_REQUESTS)
            
            # Total Rentals, 두 주자장의 최대 렌탈 찻수
            # 만약 렌탈의 갯수가 현재 차량보다 많으면 carsLoc1 만큼으로 보정
            totalRentalsLoc1 = min(carsLoc1, rentalsLoc1)
            totalRentalsLoc2 = min(carsLoc2, rentalsLoc2)
            
            # Total Rewards, 렌탈 차량 갯수에 따른 보상 값 (1주차장 렌탈 수 + 2주차장 렌탈 수) * 차량 1대 렌탈 가격
            rewards = (totalRentalsLoc1 + totalRentalsLoc2) * RENTAL_CREDIT
            
            # 차량 회수 확률 반복
            for returnsLoc1 in range(0, POISSON_UPPER_BOUND):
                for returnsLoc2 in range(0, POISSON_UPPER_BOUND):
                    
                    # Return Rate Probabilities
                    # 4사건(Request1,Request2, Returun1, Return2)이 동시에 일어날 확률 이므로 Multiply
                    prob = rentalsProb * poisson(returnsLoc1, EXPECTED_FIRST_LOC_RETURNS) * poisson(returnsLoc2, EXPECTED_SECOND_LOC_RETURNS) 
                    
                    # 차량 회수 후의 State
                    # 현재 주차장에 남아있는 차량 - 렌탈한 차량 + 회수한 차량 = 다음 날 1주차장, 2주차장 차량 수
                    carsLoc1_prime = min(carsLoc1 - totalRentalsLoc1 + returnsLoc1, MAX_CARS)
                    carsLoc2_prime = min(carsLoc2 - totalRentalsLoc2 + returnsLoc2, MAX_CARS)
                    
                    # Number of cars at the end of the day
                    # Value Iteration에 근거 Probablity * (Reward + DiscountFacetor * StateValueFunction_Prime)
                    # 기댓값 Reward = probablity * (Reward + DiscountFacetor * state_value_fucntion[다음 날 1주차장 차량 수, 다음 날 2주차장 차량 수])   
                    returns += prob * (rewards + DISCOUNT_RATE * stateValue[carsLoc1_prime, carsLoc2_prime])
                    
    return returns

#푸아송 랜덤 결과의 MAX값 지정
POISSON_UPPER_BOUND = 11       

#주차장 당 최대 보유 가능한 차
MAX_CARS = 20

#한번 옮길때 최대 이동 가능한 차
MAX_MOVE_OF_CARS = 5

#첫번째 주차장의 차량 요청률 : 푸아송 랜덤 프로세스 람다 3
EXPECTED_FIRST_LOC_REQUESTS = 3

#두번째 주차장의 차량 요청률 : 푸아송 랜덤 프로세스 람다 4
EXPECTED_SECOND_LOC_REQUESTS = 4

#첫번째 주차장의 차량 회수률 : 푸아송 랜덤 프로세스 람다 3
EXPECTED_FIRST_LOC_RETURNS = 3

#두번째 주차장의 차량 회수률 : 푸아송 랜덤 프로세스 람다 3
EXPECTED_SECOND_LOC_RETURNS = 2

#Discount Factor 0.9
DISCOUNT_RATE = 0.9

#차량 한대 렌탈 비용 10
RENTAL_CREDIT = 10

#차량 이동 시킬때 비용 2
COST_OF_MOVING = 2

#정책 각 상태에서 다음 상태로 가는 Probablity 행렬 
policy = np.zeros((MAX_CARS + 1, MAX_CARS + 1))

#State value function
state_value_function = np.zeros((MAX_CARS + 1, MAX_CARS + 1))

#State Pair list <주차장 A의 차량 수, 주차장 B의 차량 수>
states = []
for i in range(MAX_CARS + 1):
    for j in range(MAX_CARS + 1):
        states.append([i, j])
        
#행동 할 수 있는 Action 선정 -5 -> A주차장에서 B주차장으로 차량이 5대 이동, 5 -> B주차장에서 A주차장으로 차량이 5대 이동
actions = np.arange(-MAX_MOVE_OF_CARS, MAX_MOVE_OF_CARS + 1) 

new_state_value_fuction = np.zeros((MAX_CARS + 1, MAX_CARS + 1))

#State Value function iteration

iteration_count = 0
for k in range(1000):
    
    iteration_count += 1
    for i, j in states:
        new_state_value_fuction[i, j] = expectedReturn([i, j], policy[i, j], state_value_function)        
        
    diff_average = abs(np.average(abs(state_value_function) - abs(new_state_value_fuction)))
    print("step : " + str(iteration_count) +", diff average : " + str(diff_average))

    #Value function iteration의 결과 차이 총합이 0.001 보다 작으면 Iteration 끝
    if diff_average < 0.01:
        break       
                
    state_value_function = new_state_value_fuction.copy()
            
"""
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

X = []
Y = []
Z = []

for i in range(21):
    for j in range(21):
        X.append(j)
        
for i in range(21):
    for j in range(21):
        Y.append(i)
        
for i in range(21):
    for j in range(21):
        Z.append(state_value_function[i][j])
        
        
for m, zlow, zhigh in [('o', -50, -25)]:
    xs = X
    ys = Y
    zs = Z
    ax.scatter(xs, ys, zs, marker=m)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()


plt.subplot(121)
CS = plt.contour(solver.policy, levels=range(-6, 6))
plt.clabel(CS)
plt.xlim([0, 20])
plt.ylim([0, 20])
plt.axis('equal')
plt.xticks(range(21))
plt.yticks(range(21))
plt.grid('on')
      "   


print(state_value_function)
          
plt.subplot(121)
CS = plt.contour(policy, levels=range(-6, 6))
plt.clabel(CS)
plt.xlim([0, 20])
plt.ylim([0, 20])
plt.axis('equal')
plt.xticks(range(21))
plt.yticks(range(21))
plt.grid('on')
"""
print("iteration count" + str(iteration_count))
plt.subplot(122)
plt.pcolor(state_value_function)
plt.colorbar()
plt.axis('equal')

plt.show()        
    

        
       

