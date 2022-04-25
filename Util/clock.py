# -*- coding: utf-8 -*-
"""
Created on Sat May  8 22:04:58 2021

@author: Handonghee
"""
import math 
import matplotlib.pyplot as plt
import numpy as np
#반지름이 5인 시계
watchsize = 5

#초침의 시작점은 00시00분00초 시작점
#degree는 90도
startRadian = math.pi / 2 

x = np.linspace(-2, 0, 2000)
time = np.linspace(60, 0, 2000)
y = []

for addRadian in x:
    y.append(watchsize * math.cos((math.pi * addRadian) + startRadian))

plt.plot(time,y)