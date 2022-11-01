# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 13:59:28 2021

@author: CNL-B3
"""
import numpy as np
import matplotlib.pyplot as plt

def LTEPathLoss_LOS(d,fc):
    return 22.7 * np.log10(d) + 41 + 20 * np.log10(fc/5)

def NRPathLoss_LOS(d, fc):
    return 38.77 * np.log10(d) + 18.2 * np.log10(fc)

distances = np.arange(5,1000, 1)

fc_LTE = 2
fc_NR = 5.7

pathLoss_LTEs = []
pathLoss_NRs = []

for d in distances:
    pathLoss_LTEs.append(LTEPathLoss_LOS(d,fc_LTE))
    pathLoss_NRs.append(NRPathLoss_LOS(d,fc_NR))


plt.plot(pathLoss_LTEs, color='r')
plt.plot(pathLoss_NRs, color='b')