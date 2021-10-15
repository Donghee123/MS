# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 16:05:56 2020

이동 통신 시스템 HW2
jakes model

need
conda install numpy
conda install matplotlib
"""

import math
import numpy as np
from matplotlib import pyplot as plt

#Common functions
def pow2db(x):
    return 10 * np.log10(x)

def pow2dbm(x):
    return (10 * np.log10(x)) + 30

#Jakes
def Jakes(fd, Ts, Ns, t0, E0, phi_N):
    """
    Inputs:
       fd : Doppler frequency 
       Ts : sampling time
       Ns : number of samples
       t0 : initial time
       E0 : channel power
       phi_N : initial phase
    Outputs:
       h  : complex vector
    """
    N0 = 8           # As suggested by Jakes
    N= 4 * N0 + 2       # an accurate approximation
    wd = 2 * math.pi * fd   # maximum doppler frequency [rad]

    t = t0 + np.arange(0, Ns, 1).reshape(1,Ns) * Ts
    tf = t[-1] + Ts

    time = np.arange(1, (N0 +1), 1)
    
    #8개의 scattereres 형성
    scatterers = np.array([[1],[2],[3],[4],[5],[6],[7],[8]]) #same as M scatterers 


    y1 = math.sqrt(2) * np.cos(wd*t)
    y2 = 2 * np.cos(wd * np.dot(np.cos(2 * math.pi / N * scatterers) , t))
    y = np.concatenate((y1,y2), axis=0)

    x1 = E0 / math.sqrt(2 * N0 + 1) * np.exp(1j * np.array([phi_N]).reshape(1,1))
    x2 = E0 / math.sqrt(2 * N0 + 1) * np.exp(1j * np.array([math.pi / (N0+1) * time]))
    x = np.concatenate((x1,x2), axis=1)
    h = np.dot(x,y)

    return h


# Main
fd = 100       # Doppler frequency fd
Ts = 1e-6      # Sampling time, 1msec 
Ns = 500000    # Number of samples, 50,000 0000  
t0 = 0        # Initial time
E0 = 1 # Transmitted channel power, 
phi_N = 0

# Simuation 100Hz
h = Jakes(fd, Ts, Ns, t0, E0, phi_N)     # Use Jakes model to generate the complex fading vector
hMag = pow2db(abs(h))              # Magnitude of fading (dB)
hPwr = pow2db(abs(h) ** 2)           # Power of fading (dB)

yaxis1 = hPwr.T

fd = 926       # Doppler frequency fd

# Simuation 900Hz
h = Jakes(fd, Ts, Ns, t0, E0, phi_N)     # Use Jakes model to generate the complex fading vector
hMag = pow2db(abs(h))              # Magnitude of fading (dB)
hPwr = pow2db(abs(h) ** 2)           # Power of fading (dB)

#xaxis = xaxis.reshape(1,Ns)
yaxis2 = hPwr.T

#Plot 
xaxis = np.arange(1, Ns+1, 1) * Ts

fig = plt.figure(1)
ax = fig.add_subplot(111)
#ax.plot(xaxis,yaxis1, '-', label = 'Doppler freq 100Hz (dB)', linewidth=0.5)
ax.plot(xaxis,yaxis2, '-r', label = 'Doppler freq 900Hz (dB)', linewidth=0.5)


fig.legend(loc="upper right")

ax.set_xlabel("time (s)")
ax.set_ylabel("power (dB)")

ax.set_xlim(0,xaxis[-1])

fig = plt.figure(2)
ax = fig.add_subplot(111)
amplitude = abs(h).T
plt.hist(amplitude, bins=100, density=True)
plt.xlim(min(amplitude), max(amplitude))
ax.set_xlabel("amplitude |h|")
ax.set_ylabel("density")


plt.show()
 