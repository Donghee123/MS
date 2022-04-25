# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 23:34:55 2021

@author: Donghee Han
Pathloss model

simulation LOS
simulation NLOS
simulation NLOSv
"""

import numpy as np
import random 
import matplotlib.pyplot as plt

listOfLOSPathloss = []
listOfNLOSPathloss = []
listOfProbNLOSPathloss = []
listOfaboveLOSPathloss = []
listOfaboveNLOSPathloss = []
listOfaboveProbNLOSPathloss = []

belownoise = 9
abovenoise = 13

d = np.linspace(5,1000,500)
carrierBelowFrequency = 6 * (10**9)
carrierAboveFrequency = 30 * (10**9)
sampleCount = 50

def LOS(d, fc, enableBlockByVehicle = False):
    if enableBlockByVehicle == False:
        return 38.77 + 16.7 * np.log10(d) + 18.2 * np.log10(fc)
    else:
        probLOS = min(1.0, 1.05 * np.exp(-0.0114 * float(d)))
        
        valueRandom = random.random()
        
        if valueRandom <= probLOS:
            return 38.77 + 16.7 * np.log10(d) + 18.2 * np.log10(fc)
        else:
            randomData = max(0,np.random.normal(0.0, 4.0, size=1))
            addloss = 5 + max(0, 15 * np.log10(randomData) - 41) 
            return 38.77 + 16.7 * np.log10(d) + 18.2 * np.log10(fc) + addloss

def NLOS(d, fc):     
    return 36.85 + 30 * np.log10(d) + 18.9 * np.log10(fc)


for distance in d:
    
    listOfSampleLOSPathloss=[]
    listOfSampleNLOSPathloss=[]
    listOfSampleProbNLOSPathloss=[]
    
    listOfSampleaboveLOSPathloss=[]
    listOfSampleaboveNLOSPathloss=[]
    listOfSampleaboveProbNLOSPathloss=[]
                                           
    for sampling in range(sampleCount):
        listOfSampleLOSPathloss.append(LOS(distance, carrierBelowFrequency) + np.random.normal(0.0, belownoise, size=1))
        listOfSampleNLOSPathloss.append(NLOS(distance, carrierBelowFrequency) + np.random.normal(0.0, belownoise, size=1))
        listOfSampleProbNLOSPathloss.append(LOS(distance, carrierBelowFrequency, enableBlockByVehicle = True) + np.random.normal(0.0, belownoise, size=1))
        
        listOfSampleaboveLOSPathloss.append(LOS(distance, carrierAboveFrequency) + np.random.normal(0.0, abovenoise, size=1))
        listOfSampleaboveNLOSPathloss.append(NLOS(distance, carrierAboveFrequency)+ np.random.normal(0.0, abovenoise, size=1))
        listOfSampleaboveProbNLOSPathloss.append(LOS(distance, carrierAboveFrequency, enableBlockByVehicle = True) + np.random.normal(0.0, abovenoise, size=1))

    listOfLOSPathloss.append(np.mean(listOfSampleLOSPathloss))
    listOfNLOSPathloss.append(np.mean(listOfSampleNLOSPathloss))
    listOfProbNLOSPathloss.append(np.mean(listOfSampleProbNLOSPathloss))
    
    listOfaboveLOSPathloss.append(np.mean(listOfSampleaboveLOSPathloss))
    listOfaboveNLOSPathloss.append(np.mean(listOfSampleaboveNLOSPathloss))
    listOfaboveProbNLOSPathloss.append(np.mean(listOfSampleaboveProbNLOSPathloss)) 

    
plt.xlabel('distance between tx-rx (meter)')
plt.ylabel('pathloss(dB)')

plt.plot(d, listOfLOSPathloss, '.', label = 'LOS(6GHz)')
plt.plot(d, listOfNLOSPathloss,  '.', label = 'NLOS(6GHz)')
plt.plot(d, listOfProbNLOSPathloss, '.', label = 'NLOSv(6GHz)')
plt.plot(d, listOfaboveLOSPathloss, '.', label = 'LOS(30GHz)')
plt.plot(d, listOfaboveNLOSPathloss,  '.', label = 'NLOS(30GHz)')
plt.plot(d, listOfaboveProbNLOSPathloss, '.', label = 'NLOSv(30GHz)')



plt.legend()