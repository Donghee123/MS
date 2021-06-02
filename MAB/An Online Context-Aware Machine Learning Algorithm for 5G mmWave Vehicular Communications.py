# -*- coding: utf-8 -*-
"""
Created on Thu May 20 16:58:50 2021

@author: Handonghee
@Discription : 
    subject  : An Online context-Aware Machine Learning Algorithm for 5G mmWave Vehicular Commnunication
    method   : MAB
"""
import numpy as np
import matplotlib.pylab as plt
import random

#random으로 차량 생성
def CreateRandomVehiclePosition(MAP, vehicleMAX):
    trafficroads = []
    
    for i in range(len(MAP)):
        for j in range(len(MAP[0])):
            if(MAP[i][j] == 0):
                trafficroads.append((i,j))
    
    #랜덤 뽑기에서 문제
    vehiclePositions = np.random.choice(trafficroads, vehicleMAX,False)
    
    for i in vehiclePositions:
        MAP[i[0]][i[1]] = 2
    
    return MAP
    
        
#도로 생성 함수 (진입 못하는 도로 : -1, 도로 : 0,  BasetStation : 1, 차량 : 2)
def CreateMAP(width, height, roads, basestationposition):
    MAP = []
    
    #지도 생성
    for i in range(height):
        temp = []
        for j in range(width):
            temp.append(-1)
        
        MAP.append(temp)
        
    #도로 생성
    for road in roads:
        
        increaseHorizenStep= (road[0][1] - road[1][1]) / width
        increaseVerticalStep  = (road[0][0] - road[1][0]) / height
        
        #세로방향 도로 
        decistionRoadDirection = 0
        
        #가로방향 도로
        if abs(road[0][0] - road[1][0]) < abs(road[0][1] - road[1][1]):
            decistionRoadDirection = 1
        
        curpositionY = road[0][0]
        curpositionX = road[0][1]
        
        
        MAP[int(curpositionY)][int(curpositionX)] = 0
        
        for i in range(road[0][decistionRoadDirection], road[1][decistionRoadDirection] - 1):
            curpositionX += increaseHorizenStep
            curpositionY += increaseVerticalStep
            MAP[int(curpositionY)][int(curpositionX)] = 0
        
    MAP[int(basestationposition[0])][int(basestationposition[1])] = 1
    return np.array(MAP)
    
            
#시물레이션 맵 가로
MAP_WIDTH = 50
#시뮬레이션 맵 세로 크기

MAP_HEIGHT = 50

#현재 Map에 차량이 있을 수 
VEHICLE_MAX = 4

#BaseStation Position
MM_BASESTATION_POSITION = [25,25]

#Beam의 방사각
MM_BASESTATION_BEAM_WDITH = 30

#한번에 선택가능한 최대 beam의 갯수 m
SELECT_MAX = 3

#도로 설정 가로축 길 1개, 세로축 길 1개
TRAFFIC_ROADS = [[(0,35),(MAP_HEIGHT, 35)], [(35,0),(35, MAP_WIDTH)], 
                 [(15,0),(15, MAP_WIDTH)], [(0,10),(MAP_HEIGHT, 10)]]

MAP = CreateMAP(width=MAP_WIDTH, height=MAP_HEIGHT,basestationposition=MM_BASESTATION_POSITION,roads=TRAFFIC_ROADS)
MAP = CreateRandomVehiclePosition(MAP, VEHICLE_MAX)   

plt.imshow(MAP, interpolation='nearest', cmap=plt.cm.bone_r)





