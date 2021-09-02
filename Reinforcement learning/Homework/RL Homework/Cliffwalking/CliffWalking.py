# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 12:08:31 2021

@author: Handonghee
"""

import numpy as np
import matplotlib.pyplot as plt
import time

# world height
MAX_HEIGHT = 4

# world width
MAX_WIDTH = 12

#CliffState
CliffMap = [1,2,3,4,5,6,7,8,9,10]

#Init QMap, 10x[[상,하,좌,우] x 10]
QMap = np.zeros((MAX_HEIGHT, MAX_WIDTH, 4))
Q_learningMap = np.zeros((MAX_HEIGHT, MAX_WIDTH, 4))

actions = [0,1,2,3]
#state position
start_position = [3,0]
#terminal position
goal_position = [3,11]

#Epsilon
# 0 : random, 1 : grid
e = [0.9,0.1]

#difcountfacetor
discount = 1
#gamma
gamma = 0.5

eposidestep = 500
loop = 50

Q_allreward = np.zeros(eposidestep)
Q_episodepertimesteps = np.zeros(eposidestep)
sarsa_episodepertimesteps = np.zeros(eposidestep)
sarsa_allreward = np.zeros(eposidestep)

def getNextstep(state, action):
    i, j = state
    if action == 0: #Up
        return [max(i - 1, 0), j], -1
    elif action == 1: #Down
        nextstate = [min(i + 1, MAX_HEIGHT - 1), j]
 
        if nextstate[0] == 3:
            if CliffMap[0]<=nextstate[1]<=CliffMap[-1]:
               nextstate = start_position.copy()
               
        return nextstate, -100
    
    elif action == 2: #left
        return [i, max(j - 1, 0)], -1

    elif action == 3: #right
        nextstate =  [i, min(j + 1, MAX_WIDTH - 1)]
       
        if nextstate[0] == 3:
            if CliffMap[0]<=nextstate[1]<=CliffMap[-1]:
               nextstate = start_position.copy()
               
        return nextstate, -100
    else:
        assert False

def displayoptimalpolicyresult(methodname, QMap):
    
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
    print(methodname + ' Optimal policy')
    for row in optimal_policy:
        print(row)
       


for i in range(loop):    
    #Sarsa iteration
    
    
    
    sarsa_sumepisodic = []
    sarsa_alltiemstep = 1
    
    for episode in range(eposidestep):
        sumreward = 0
        timestep = 1
        curposition = start_position.copy()
        
        choicemethod = np.random.choice(2,1,p=e)
                           
        if choicemethod == 1:  #random   
            action = np.random.choice(actions)
        else:#grid
            values_ = QMap[curposition[0], curposition[1], :]
            action = np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])
        
        while(True):
            sarsa_alltiemstep += 1
            nextposition,reward = getNextstep(curposition, action) 
            sumreward += reward
            timestep += 1    
                  
            #choice Epsilon
            choicemethod = np.random.choice(2,1,p=e)
           
            if choicemethod == 1:  #random        
                next_action = np.random.choice(actions)                  
            else: #grid
                values_ = QMap[nextposition[0], nextposition[1], :]
                next_action = np.random.choice([action_ for action_, \
                                                value_ in enumerate(values_) if value_ == np.max(values_)])
            
            target = QMap[nextposition[0], nextposition[1], next_action]
       
            target *= discount
            
            # Sarsa update 
            QMap[curposition[0], curposition[1], action] += gamma * (reward + target - QMap[curposition[0], curposition[1], action])
        
            curposition = nextposition
            action = next_action
            sarsa_sumepisodic.append([sarsa_alltiemstep, episode])
            
            if curposition[0] is goal_position[0] and curposition[1] is goal_position[1]:
                break
        
        sarsa_allreward[episode] += (sumreward)            
        sarsa_episodepertimesteps[episode] += (timestep)
        
    #Q-learning iteration

    Q_sumepisodic = []
    Q_alltiemstep = 1

   
    for episode in range(eposidestep):
        sumreward = 0
        timestep = 1
        curposition = start_position.copy()
                                           
        while(True):
            
            choicemethod = np.random.choice(2,1,p=e)    
            if choicemethod == 1:  #random   
                action = np.random.choice(actions)
            else:#grid
                values_ = Q_learningMap[curposition[0], curposition[1], :]
                action = np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])
        
            
            Q_alltiemstep += 1
            nextposition,reward = getNextstep(curposition, action) 
            sumreward += reward
            timestep += 1    
                  
              
            # Q-learning update
            Q_learningMap[curposition[0], curposition[1], action] += gamma * (
                reward + discount * np.max(Q_learningMap[nextposition[0], nextposition[1], :]) -
                Q_learningMap[curposition[0], curposition[1], action])
            

            curposition = nextposition
            action = next_action
            Q_sumepisodic.append([Q_alltiemstep, episode])
            
            if curposition[0] is goal_position[0] and curposition[1] is goal_position[1]:
                break
        
        Q_allreward[episode] += sumreward            
        Q_episodepertimesteps[episode] += timestep
    

Q_allreward /= loop
sarsa_allreward /= loop

# draw reward curves
plt.plot(sarsa_allreward, label='Sarsa')
plt.plot(Q_allreward, label='Q-Learning')
plt.xlabel('Episodes')
plt.ylabel('Sum of rewards during episode')
plt.ylim([-100, 0])

displayoptimalpolicyresult('sarsa',QMap)
displayoptimalpolicyresult('q-learning',Q_learningMap)