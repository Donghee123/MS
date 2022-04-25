# -*- coding: utf-8 -*-
"""
Spyder Editor

이동 통신 시스템 HW2
lognormal shadowing

need
conda install numpy
conda install matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

def GetLognormal(mu, sigma, sampleSize:int):
    return np.random.lognormal(mu, sigma, sampleSize)
    
    
sample = GetLognormal(3,0.5,10000)
count, bins, ignored = plt.hist(sample, 500, density=True, align='mid')



