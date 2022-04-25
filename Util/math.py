# -*- coding: utf-8 -*-
"""
Created on Thu May  6 17:30:31 2021

@author: CNL-B3
"""
import math 
import matplotlib.pyplot as plt
import numpy as np

def sinc(x):
    if x == 0:
        return 1
    else:
        return (math.sin(math.pi * x) / (math.pi * x))

x = np.linspace(-5,5,1000)
y = []

for valueX in x:
    y.append(sinc(valueX))


plt.plot(x,y)
plt.show()