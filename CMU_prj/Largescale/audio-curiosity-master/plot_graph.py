from socket import TCP_FASTOPEN
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import pandas as pd

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

f1 = open('04-19-11-54-14__Breakout_fft_128_True_True_data.csv', 'r') #기본 feat dim : 256 
f2 = open('04-20-20-31-52__Breakout_fft_128_True_True_data.csv', 'r') #기본 feat dim : 512

f3 = open('04-18-10-55-41__Breakout_fft_128_True_True_data.csv', 'r') #VAE 512
f4 = open('04-19-11-33-59__Breakout_fft_128_True_True_data.csv', 'r') #VAE feat dim : 256

f5 = open('extrinsic_intrinsic_reward_method_data.csv', 'r') #reward merge

df1 = pd.read_csv(f1)
df2 = pd.read_csv(f2)

df3 = pd.read_csv(f3)
df4 = pd.read_csv(f4)

df5 = pd.read_csv(f5)

x = 't_step'
y = 'reward_mean'

x1 = df1[x]
y1 = df1[y]

x_smooth = 111
y_smooth = 7
box_smooth = 101

yhat1 = savgol_filter(y1, x_smooth, y_smooth) 


x2 = df2[x]
y2 = df2[y]

yhat2 = savgol_filter(y2, x_smooth, y_smooth) 


x3 = df3[x]
y3 = df3[y]

yhat3 = savgol_filter(y3, x_smooth, y_smooth) 

x4 = df4[x]
y4 = df4[y]

yhat4 = savgol_filter(y4, x_smooth, y_smooth)

x5 = df5[x]
y5 = df5[y]

yhat5 = savgol_filter(y5, x_smooth, y_smooth)

plt.plot(x1,y1, color='green', label='baseline fd 256', alpha=0.2)
plt.plot(x2,y2, color='red', label='baseline fd 512', alpha=0.2)
plt.plot(x1,yhat1, color='green', label='baseline fd 256 smooth')
plt.plot(x2,yhat2, color='red', label='baseline fd 512 smooth')

plt.plot(x3,y3, color='blue', label='VAE fd 512', alpha=0.2)
plt.plot(x4,y4, color='black', label='VAE fd 256', alpha=0.2)
plt.plot(x3,yhat3, color='blue', label='VAE fd 512 smooth')
plt.plot(x4,yhat4, color='black', label='VAE fd 256 smooth')

plt.plot(x5,y5, color='cyan', label='reward(ex + in) fd 256', alpha=0.2)
plt.plot(x5,yhat5, color='cyan', label='reward(ex + in) fd 256 smooth')

plt.xlabel('step')
plt.ylabel('mean extrinsic reward')
plt.legend()
plt.show()