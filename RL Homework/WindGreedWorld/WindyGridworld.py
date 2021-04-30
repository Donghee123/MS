# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 12:08:31 2021

@author: Handonghee
"""

import numpy as np
import matplotlib.pyplot as plt
import time

# world height
MAX_HEIGHT = 7

# world width
MAX_WIDTH = 10

#WindState
WindMap = [0,0,0,1,1,1,2,2,1,0]
   
def getNextstep(state, action):
    i, j = state
    if action == 0: #Up
        return [max(i - 1 - WindMap[j], 0), j]
    elif action == 1: #Down
        return [max(min(i + 1 - WindMap[j], MAX_HEIGHT - 1), 0), j]
    elif action == 2: #left
        return [max(i - WindMap[j], 0), max(j - 1, 0)]
    elif action == 3: #right
        return [max(i - WindMap[j], 0), min(j + 1, MAX_WIDTH - 1)]
    else:
        assert False

def displayoptimalpolicyresult():
    
    # display the optimal policy
    optimal_policy = []
    for i in range(0, MAX_HEIGHT):
        optimal_policy.append([])
        for j in range(0, MAX_WIDTH):
            if [i, j] == goal_position:
                optimal_policy[-1].append('G')
                continue
            bestAction = np.argmax(QMap[i, j, :])
            if bestAction == 0:
                optimal_policy[-1].append('U')
            elif bestAction == 1:
                optimal_policy[-1].append('D')
            elif bestAction == 2:
                optimal_policy[-1].append('L')
            elif bestAction == 3:
                optimal_policy[-1].append('R')
    print('Optimal policy')
    for row in optimal_policy:
        print(row)
       




#Init QMap, 10x[[상,하,좌,우] x 10]
QMap = np.zeros((MAX_HEIGHT, MAX_WIDTH, 4))

actions = [0,1,2,3]
#state position
start_position = [3,0]
#terminal position
goal_position = [3,7]

#Epsilon
# 0 : random, 1 : grid
e = [0.9,0.1]

#gamma
gamma = 0.5
#np.random.choice(4,1,p=directions)
 
#iteration
episodepertimesteps = []

sumepisodic = []
alltiemstep = 1
for episode in range(500):
    
    timestep = 1
    curposition = start_position.copy()
    
    choicemethod = np.random.choice(2,1,p=e)
    
    curMovedirection = [0,0]
                
    if choicemethod == 1:  #random   
        action = np.random.choice(actions)
    else:#grid
        values_ = QMap[curposition[0], curposition[1], :]
        action = np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])
    
    while(True):
        alltiemstep += 1
        nextposition = getNextstep(curposition, action)       
        timestep += 1    
              
        #choice Epsilon
        choicemethod = np.random.choice(2,1,p=e)
       
        if choicemethod == 1:  #random        
            next_action = np.random.choice(actions)                  
        else: #grid
            values_ = QMap[nextposition[0], nextposition[1], :]
            next_action = np.random.choice([action_ for action_, \
                                            value_ in enumerate(values_) if value_ == np.max(values_)])
        
        reward = -1
        # Sarsa update
        QMap[curposition[0], curposition[1], action] += \
            gamma * (reward + QMap[nextposition[0], nextposition[1], next_action] -
                     QMap[curposition[0], curposition[1], action])
            
        curposition = nextposition
        action = next_action
        sumepisodic.append([alltiemstep, episode])
        
        if curposition[0] is goal_position[0] and curposition[1] is goal_position[1]:
            break
    
                
    episodepertimesteps.append(timestep)

x = []
for i in sumepisodic:
   x.append(i[0]) 
   
y = []
for i in sumepisodic:
   y.append(i[1]) 

plt.plot(x,y)
displayoptimalpolicyresult()