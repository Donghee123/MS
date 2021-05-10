# -*- coding: utf-8 -*-
"""
Created on Mon May 10 23:58:42 2021

@author: Handonghee
"""

"""
Created on Sat May  8 22:04:58 2021

@author: Handonghee
"""
import math 
import matplotlib.pyplot as plt
import numpy as np
import random 

#-0.5~0.5 사이의 랜덤 값 추출
def makeNoise():
    if (random.random()>0.5):
        return random.random() / 2
    else:
        return -random.random() / 2

#증폭 1
amplitude = 1

#위상차이 0
phaseshift =0


x = np.linspace(0, 0.5, 200)
y = []

for addRadian in x:
    y.append((amplitude * math.cos((math.pi * addRadian) + phaseshift)) + makeNoise())
    #y.append((amplitude * math.cos((math.pi * addRadian) + phaseshift)))

plt.plot(x,y)