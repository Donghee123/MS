# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 13:30:38 2021

@author: CNL-B3
"""
import pandas as pd
import numpy as np
import time
import random
import math
import matplotlib.pylab as plt
import matplotlib.patches as patches
from Environment import Environ as Environ

up_lanes = [3.5/2,3.5/2 + 3.5,250+3.5/2, 250+3.5+3.5/2, 500+3.5/2, 500+3.5+3.5/2]
down_lanes = [250-3.5-3.5/2,250-3.5/2,500-3.5-3.5/2,500-3.5/2,750-3.5-3.5/2,750-3.5/2]
left_lanes = [3.5/2,3.5/2 + 3.5,433+3.5/2, 433+3.5+3.5/2, 866+3.5/2, 866+3.5+3.5/2]
right_lanes = [433-3.5-3.5/2,433-3.5/2,866-3.5-3.5/2,866-3.5/2,1299-3.5-3.5/2,1299-3.5/2]

width = 750
height = 1299

v2ilink_fastfading = []

v2vlink_pathloss = []
v2ilink_pathloss = []
v2vlink_shadowing = []
v2vlink_fastfading = []
v2vlink_allfading = []
v2ilink_allfading = []

v2vlink_neibor1 = []
v2vlink_neibor2 = []
v2vlink_neibor3 = []


def SaveNumpyArrayDataToCSV(saveName, datas_numpyArray):
    
    df = pd.DataFrame(datas_numpyArray)
    df.to_csv(saveName, index=False)

def Show_V2V_ALL_lossgraph(ax, Env, pair, rbIndex, tick, color):
    
    
    ax.set_ylim(0, 200)
    ax.set_xlim(0,int(teststep/capture_term))
    ax.plot(np.array(v2vlink_allfading), color)
    
    ax.tick_params(axis='both', direction='in')
    #ax.set_xlabel('Time step({}ms)'.format(tick))
    #ax.set_ylabel('V2V loss(Db)')

def Show_V2I_ALL_lossgraph(ax, Env, vehicle_index, rbIndex, tick, color):
    
    
    ax.set_ylim(0, 200)
    ax.set_xlim(0,int(teststep/capture_term))
    ax.plot(np.array(v2ilink_allfading), color)
    
    ax.tick_params(axis='both', direction='in')
    #ax.set_xlabel('Time step({}ms)'.format(tick))
    #ax.set_ylabel('V2I loss(Db)')
    
def Show_V2V_Pathlossgraph(ax, Env, pair, tick, color):
    
    
    
    ax.set_ylim(0, 200)
    ax.set_xlim(0,int(teststep/capture_term))
   
    ax.plot(np.array(v2vlink_pathloss), color)
    ax.tick_params(axis='both', direction='in')
    #ax.set_xlabel('Time step({}ms)'.format(tick))
    #ax.set_ylabel('Pathloss(Db)')
    
def Show_Neighbor_V2V_Pathlossgraph(ax1, ax2, ax3, Env, pair, tick, rbIndex):
    
    
    ax1.set_ylim(0, 200)
    ax1.set_xlim(0,int(teststep/capture_term))
   
    ax2.set_ylim(0, 200)
    ax2.set_xlim(0,int(teststep/capture_term))
    
    ax3.set_ylim(0, 200)
    ax3.set_xlim(0,int(teststep/capture_term))
    
    ax1.plot(np.array(v2vlink_neibor1), 'b')
    ax1.tick_params(axis='both', direction='in')
    
    ax2.plot(np.array(v2vlink_neibor2), 'g')
    ax2.tick_params(axis='both', direction='in')
    
    ax3.plot(np.array(v2vlink_neibor3), 'r')
    ax3.tick_params(axis='both', direction='in')
    
    #ax.set_xlabel('Time step({}ms)'.format(tick))
    #ax.set_ylabel('Pathloss(Db)')
    
def Show_V2I_Pathlossgraph(ax, Env, pair, tick, color):
    
    
    
    ax.set_ylim(0, 200)
    ax.set_xlim(0,int(teststep/capture_term))
    
    ax.plot(np.array(v2ilink_pathloss), color)
    ax.tick_params(axis='both', direction='in')
    #ax.set_xlabel('Time step({}ms)'.format(tick))
    #ax.set_ylabel('Pathloss(Db)')
    
def Show_Shadowinggraph(ax, Env, pair, tick):
    
    
    
    ax.set_ylim(np.array(v2vlink_shadowing).min(), np.array(v2vlink_shadowing).max()+np.array(v2vlink_shadowing).max()/10)
    ax.plot(np.array(v2vlink_shadowing), 'g')
    
    #ax.set_xlabel('Time step({}ms)'.format(tick))
    #ax.set_ylabel('Shadowing(Db)')
    
def Show_Fastfading(ax, Env, pair, rbIndex, tick):
    
             
    ax.set_ylim(np.array(v2vlink_fastfading).min(), np.array(v2vlink_fastfading).max()+np.array(v2vlink_fastfading).max()/10)    
    ax.plot(np.array(v2vlink_fastfading),'r')
    
  
    ax.set_xlabel('Time step({}ms)'.format(tick))
    #ax.set_ylabel('Fast fading(Db)')

    
def Show_2Dmap_plot(ax, Env, width, height, anlysysVehiclesPair, IsAdaptneighbors = False):    
    position_BaseStation = Env.V2Ichannels.BS_position
    ax.set_ylim(-100, height +  100)
    ax.set_xlim(-100, width + 100)
    ax.tick_params(axis='both', direction='in')
    position_BaseStation
    
    ax.add_patch(
   patches.Rectangle(
      (position_BaseStation[0], position_BaseStation[1]),             
      30, 30,                    
      edgecolor = 'deeppink',
      facecolor = 'lightgray',
      fill=True,
   ))
    
    for index in range(len(Env.vehicles)):
        vehicle = Env.vehicles[index]
        
        positionX_vehicle = vehicle.position[0]
        positionY_vehicle = vehicle.position[1]  
        
        color = 'b'
        direction = [30,0]
        width = 50
        if vehicle.direction == 'u':
            color = 'g'
            direction = [0,60]
            width = 27
        elif vehicle.direction == 'd':
            color = 'r'
            direction = [0,-60]
            width = 27
        elif vehicle.direction == 'l':
            color = 'violet'
            direction = [-30,0]
        
        ax.add_patch(
        patches.Arrow(
        positionX_vehicle, positionY_vehicle,
        direction[0], direction[1],
        width=width,        
        edgecolor = color,
        facecolor = color
     ))

    firstVehicle = Env.vehicles[anlysysVehiclesPair[0]]    
    
    if IsAdaptneighbors == False:        
        secondVehicle = Env.vehicles[anlysysVehiclesPair[1]]        
        ax.plot([firstVehicle.position[0],secondVehicle.position[0]] ,[firstVehicle.position[1],secondVehicle.position[1]], 'b--')
        ax.plot([firstVehicle.position[0],position_BaseStation[0]] ,[firstVehicle.position[1],position_BaseStation[1]], 'r--')
    else:
        colors = ['b--','g--','r--']
        for i in range(len(firstVehicle.destinations)):
            connectedVehicle = Env.vehicles[firstVehicle.destinations[i]]
            ax.plot([firstVehicle.position[0],connectedVehicle.position[0]] ,[firstVehicle.position[1],connectedVehicle.position[1]], colors[i])
            
    

vehicleNumber = 20
#환경 생성
Env = Environ(down_lanes,up_lanes,left_lanes,right_lanes, width, height, vehicleNumber) 


Env.add_new_vehicles_by_number(int(vehicleNumber/4))




teststep= 50000

capture_term = 100
renew_neighbor_term = 1000 

"""
현재 상태 그래프로 보여주기
figs = []
axs = []
for i in range(int(teststep/capture_term)):
    figs.append(plt.figure(i))

for i in range(len(figs)):
    axs.append(figs[i].add_subplot(1,2,1))
    axs.append(figs[i].add_subplot(3,2,2))
    axs.append(figs[i].add_subplot(3,2,4))
    axs.append(figs[i].add_subplot(3,2,6))
"""   

neighbors_number = 3
count = 0;
time_step = 0.1
anlysysVehiclesPair = (0,2)
observeRbIndex = 0

vehicle_pair = (0,3) #측정을 원하는 차량 V2V 링크 

onetick = Env.timestep
tick = capture_term * onetick
IsAdaptneighbors = True
color = ['b','g','r']

for i in range(teststep):
    
    Env.renew_positions()
    Env.renew_channels_fastfading()    
    
    if i == 0:
        Env.renew_neighbor()
        
    """
    AddData
    """
    #if i % capture_term == 0:
        #측정을 원하는 두차량의 v2v link loss 측정
    v2v_lossValue = Env.V2V_channels_with_fastfading[vehicle_pair[0]][vehicle_pair[1]][observeRbIndex]   
    v2vlink_allfading.append(v2v_lossValue)   
        
        #측정을 원하는 차량의 v2i link loss 측정
    v2i_lossValue = Env.V2I_channels_with_fastfading[vehicle_pair[0]][observeRbIndex]        
    v2ilink_allfading.append(v2i_lossValue)
        
        #측정을 원하는 두차량의 v2v link pathloss만 측정
    pathlossValue = Env.V2Vchannels.PathLoss[vehicle_pair[0]][vehicle_pair[1]]       
    v2vlink_pathloss.append(pathlossValue)
        
        #현재 차량과 연결된 주변 차량 3대 측정
    ConnectedVehicleIndexs = Env.vehicles[vehicle_pair[0]].destinations
                
    pathlossValue1 = Env.V2V_channels_with_fastfading[vehicle_pair[0]][ConnectedVehicleIndexs[0]][observeRbIndex]
    pathlossValue2 = Env.V2V_channels_with_fastfading[vehicle_pair[0]][ConnectedVehicleIndexs[1]][observeRbIndex]
    pathlossValue3 = Env.V2V_channels_with_fastfading[vehicle_pair[0]][ConnectedVehicleIndexs[2]][observeRbIndex]
             
    v2vlink_neibor1.append(pathlossValue1)
    v2vlink_neibor2.append(pathlossValue2)
    v2vlink_neibor3.append(pathlossValue3)
            
        #측정을 원하는 차량의 v2i link pathloss만 측정
    pathlossValue = Env.V2Ichannels.PathLoss[vehicle_pair[0]]        
    v2ilink_pathloss.append(pathlossValue)
        
        #측정을 원하는 두차량의 v2v link Shadowing만 측정
    shadowing = Env.V2Vchannels.Shadow[vehicle_pair[0]][vehicle_pair[1]]    
    v2vlink_shadowing.append(shadowing)
        
        #측정을 원하는 두차량의 v2v link fastfading만 측정
    fastfadingValue = Env.V2Vchannels.FastFading[vehicle_pair[0]][vehicle_pair[1]][observeRbIndex]                                                             
    v2vlink_fastfading.append(fastfadingValue)
    
    """
    인접한 차량 중 통신할 차량 3대 재갱신
    """
    if i % renew_neighbor_term == 0:
        Env.renew_neighbor()
    
    """
    현재 상태 그래프로 보여주기
    """
    #if i % capture_term == 0:
        #Show_2Dmap_plot(axs[count], Env, width, height,anlysysVehiclesPair, IsAdaptneighbors=IsAdaptneighbors)        
        #Show_Neighbor_V2V_Pathlossgraph(axs[count+1], axs[count+2], axs[count+3], Env, anlysysVehiclesPair, tick, 0)   
        
        #Show_V2I_ALL_lossgraph(axs[count+1], Env, pair[0], observeRbIndex, tick, 'r')       
        #Show_V2V_Pathlossgraph(axs[count+2], Env, pair, tick, 'b')
        #Show_V2I_Pathlossgraph(axs[count+2], Env, pair, tick, 'b')
        #count += 4
       

neibors_NumpyArray = []
neibors_NumpyArray.append(v2vlink_neibor1)
neibors_NumpyArray.append(v2vlink_neibor2)
neibors_NumpyArray.append(v2vlink_neibor3)
neibors_NumpyArray = np.array(neibors_NumpyArray).transpose()

SaveNumpyArrayDataToCSV('neibor.csv', neibors_NumpyArray)


#Env.test_channel()

#plt.plot(v2vlink_fastfading)
#plt.plot(v2ilink_fastfading)


#plt.show()


#plt.scatter(positionX_vehicle, positionY_vehicle, color = color ,label='vehicle', marker='x')
    
#Env.test_channel()    
