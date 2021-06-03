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
import math
from enum import Enum

#방향 설정 enum class -> 추후 전문성이 고려될때 사용
class DIRECTION(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

#contextData 종류 enum
class CONTEXTDATA_NAME(Enum):
    POSITIONY = 0
    POSITIONX = 0
    VELOCITY = 1
    DIRECTION = 2

#Vehicle 클래스, 좌표, 속력, 방향 데이터를 가지고 있음. 
class Vehicle:
    def __init__(self, position, velocity, direction, map_width, map_height):
        self.direction = direction
        self.velocity = velocity
        self.position = position 
        
        #실제 위치를 context Data로 변환
        self.contextposition = (position[0] / map_height, position[1] / map_width,)
    
#Context data의 PartitialData 종류를 가지고 있음
class contextPartitialData:
    def __init__(self, dataName, partitialCount):
        self.dataName = dataName
        self.partitialStep = 1 / partitialCount
    
    def GetContextPartitalData(self, dataName, continousData):
        
        #입력 데이터의 타입이 다르면 이상한 값으로 인식 -1리턴 또는 contextData가 1을 넘을 경우 이상한 값으로 추론
        if self.dataName != dataName or continousData > 1:
            return -1
                
        PartitialData = 0
        step = 0
        
        while(True):
            PartitialData += self.partitialStep
            if PartitialData > continousData:
                return step
            else:
                step += 1
        
        
class mmBaseStation:
    def __init__(self, MAP ,position, beamWidth, beamCount, contextDataList, selectMax, expolitationControl):
        self.MAP = MAP
        self.position = position
        self.beamWidth = beamWidth
        self.beamCount = beamCount
        self.contextDataList = contextDataList
        self.selectMax = selectMax
        self.expolitationControl = expolitationControl
        
        #Key : context info, Item : Expected Received Powers list 
        self.BanditsExpected = {}
        
        #Key : context info, Item : Selected Counts list
        self.BanditsCount = {}
    
    #맵에서 보이는 차량들의 정보를 읽어오고 내부적으로 처리할 수 있도록 하기
    def ConvertVehiclesContext(self, vehicles):
        contextDatas = []
                
        for vehicle in vehicles:
            
            #차량 1대의 contextData 취합, 지금은 position으로만 선택함
            contextTemp = []
            
            for context in self.contextDataList:
                if context.dataName == CONTEXTDATA_NAME.POSITIONY:
                    contextTemp.append(context.GetContextPartitalData(context.dataName, vehicle.contextposition[0]))
                elif context.dataName == CONTEXTDATA_NAME.POSITIONX:
                    contextTemp.append(context.GetContextPartitalData(context.dataName, vehicle.contextposition[1]))
            
            contextDatas.append(contextTemp)
        
        return contextDatas
    
    #List 값을 문자열로 변환 -> Bandits의 key값으로 변환 하기 위함
    def ConvertListtoString(self, lists):
        convertState = ""
        
        for i in lists:
            convertState += str(int(i)) + ','
            
        return convertState
    
    #차량과의 거리 계산
    def GetVehicleDistance(self, vehicle):
        diff_y = abs(self.position[0] - vehicle.position[0]) 
        diff_x = abs(self.position[1] - vehicle.position[1])       
        return math.sqrt(pow(diff_y,diff_y) + pow(diff_x, diff_x))
    
    def SelectGreedyBeams(self,contextKey_indexList_exploitation_Dict, selectCount):
               
        ServiceContext_beam = []
        
        while(True):
            
            if len(ServiceContext_beam) == selectCount:
                break
                
                greedyKey = ''
                greedyBeamIndex = -1
                greedyExpectedValue = -1
                
                #해당 컨텍스트에서 최대 값 찾기
                for key, items in contextKey_indexList_exploitation_Dict.items():
                    for beamIndex in items:
                        beamExpectedValue = self.BanditsExpected[key][beamIndex]
                        if beamExpectedValue > greedyExpectedValue:
                            greedyExpectedValue = beamExpectedValue
                            greedyKey = key
                            greedyBeamIndex = beamIndex
                            
                ServiceContext_beam.append((greedyKey, greedyBeamIndex))
                
                
        return ServiceContext_beam
                
    #빔 포밍할 차량 선택 bandit의 상태에 따라 exploration, exploitation중 선택됨.
    def Action(self,Vehicles):
        
        #차량별로 contextData를 가져옴
        VehiclesContextDatas = self.ConvertVehiclesContext(Vehicles)
        
        #context별 방문 상태 체크
        context_beamindex_exploration_Pairs = []
        context_beamindex_exploition_Pairs = []
        for vehiclesContext in VehiclesContextDatas:
            
            banditskey = self.ConvertListtoString(vehiclesContext)
            
            #방문 횟수 리스트
            banditsCount = self.GetBanditCountValues(banditskey)
            
            #해당 컨텍스트의 선택 했던 빔들의 결과
            banditsExpected = self.GetBanditExpectedValues(banditskey)
                       
            for index in range(len(banditsCount)):
                
                #충분한 시행을 하지 않은 것들은 context_beamindex_exploration_Pairs에 저장
                if banditsCount[index] < self.expolitationControl:
                    context_beamindex_exploration_Pairs.append((banditskey, index))
                else:#충분한 시행을 겪은 것들은 context_beamindex_exploition_Pairs에 저장
                    context_beamindex_exploition_Pairs.append((banditskey, index))
                    
        #key값 당 exploration 해야하는 액션인덱스들만 모음
        contextKey_ValueList_exploration_Dict = {}
        
        for i in context_beamindex_exploration_Pairs:
            if contextKey_ValueList_exploration_Dict.get(i[0]) == None:
                contextKey_ValueList_exploration_Dict[i[0]] = [i[1]]
            else:
                contextKey_ValueList_exploration_Dict[i[0]].append(i[1])
        
        #key값 당 exploitation 해야하는 액션인덱스들만 모음
        contextKey_ValueList_exploitation_Dict = {}
        
        for i in context_beamindex_exploition_Pairs:
            if contextKey_ValueList_exploitation_Dict.get(i[0]) == None:
                contextKey_ValueList_exploitation_Dict[i[0]] = [i[1]]
            else:
                contextKey_ValueList_exploitation_Dict[i[0]].append(i[1])
                
                
        u = len(contextKey_ValueList_exploration_Dict)
        
        #아래의 모든 과정을 거친후 서비스 할 Context에 따른 beam
        ServiceContext_beam = []
        
        #exploitation 수행 모두 greedy하게 수행
        if u == 0:
            ServiceContext_beam = self.SelectGreedyBeams(contextKey_ValueList_exploitation_Dict, self.selectMax)            
        #exploration 수행
        else:           
            #exploration이 필요한 context가 한번에 선택할 것들중에 적을 경우 
            if u < self.selectMax:
                
                #u개 만큼 random 액션 선택
                for key, items in contextKey_ValueList_exploration_Dict.items():
                    ServiceContext_beam.append((key,random.choice(items)))
                 
                #남은 동시 선택 가능한 값 그리디로 찾기                                
                while(True):
                    
                    #서비스할 빔을 다 선택했을 경우
                    if len(ServiceContext_beam) == self.selectMax:
                        break
                    
                    maxKey = ''
                    maxValueindex = -1
                    maxValue = -1
                    
                    #중복 context 배제하고 최대 값 찾기
                    for key, items in contextKey_ValueList_exploitation_Dict.items():
                        
                        isAppened = False
                        
                        #위에서 이미 선택 한 contest의 beam은 고려하지 않기 위함
                        for pair in ServiceContext_beam:
                            if pair[0] == key:
                                isAppened = True
                                break
                            
                        #위에서 이미 랜덤 선택 한 경우 패스   
                        if isAppened == True:
                            break
                        else:
                            for i in items:
                                if(maxValue > self.BanditsExpected[key][i]):
                                    maxValue = self.BanditsExpected[key][i]
                                    maxKey = key
                                    maxValueindex = i
                     
                    ServiceContext_beam.append((maxKey,maxValueindex))
                    
                  
            #self.selectMax개 만큼 모두 랜덤 선택
            else:                
                
                for key, items in contextKey_ValueList_exploration_Dict.items():
                    ServiceContext_beam.append((key,random.choice(items)))
                    
                    if len(ServiceContext_beam) == self.selectMax:
                        break
            
            #모든 선택을 마친 결과 데이터, List(contextkey, beamIndex)
            self.ExcuteServiceAndUpdate(ServiceContext_beam, Vehicles)
            
                    
    def ExcuteServiceAndUpdate(self,ServiceContext_beam, Vehicles):
        
        VehiclesContextDatas = self.ConvertVehiclesContext(Vehicles)
        VehiclesContextKeyList = []
       
        #차량의 context 데이터를 key값으로 변경 하기
        for vehiclesContext in VehiclesContextDatas:            
            VehiclesContextKeyList.append(self.ConvertListtoString(vehiclesContext))
            
        for serviceData in ServiceContext_beam:
            service_contextKey = serviceData[0]
            service_beamIndex  = serviceData[1]
            
            #선택한 인덱스의 빔을 쐈을때 선택한 컨텍스트 데이터를 가진 차량이 빔의 영역에 있는지 검사 필요, 임시 데이터로 1 저장
            receivedData = 1
            
            #Expected Value 계산 
            banditExpectedDict = self.GetBanditExpectedValues(service_contextKey)
            banditCountDict = self.GetBanditCountValues(service_contextKey)
            
            numerator = ((banditExpectedDict[service_beamIndex] * banditCountDict[service_beamIndex]) + receivedData)
            denominator  = banditCountDict[service_beamIndex] + 1
            
            #Expected Value 업데이트
            self.BanditsExpected[service_contextKey][service_beamIndex] = numerator / denominator
            self.BanditsCount[service_contextKey][service_beamIndex] = self.BanditsCount[service_contextKey][service_beamIndex] + 1
            
                       
    def GetBanditExpectedValues(self, banditskey):
        
        #BanditExpected
        if banditskey in self.BanditsExpected:
            return self.BanditsExpected[banditskey]
        else:
            self.BanditsExpected[banditskey] = [0 for _ in range(self.beamCount)]
        
        return self.BanditsExpected[banditskey] 
    
    def GetBanditCountValues(self, banditskey):        
        
        #BanditCount
        if banditskey in self.BanditsCount:
            return self.BanditsCount[banditskey]
        else:
            self.BanditsCount[banditskey] = [0 for _ in range(self.beamCount)]
            
        return self.BanditsCount[banditskey]
        
    
#random으로 차량 생성 (좌표, 속력, 방향)
def CreateRandomVehicle(MAP, vehicleMAX):
    trafficroads = []
    
    for i in range(len(MAP)):
        for j in range(len(MAP[0])):
            if(MAP[i][j] == 0):
                trafficroads.append((i,j))
    
    #차량 위치 랜덤 뽑기
    vehiclePositions =  random.sample(trafficroads, k=vehicleMAX) 
    
    #차량 속도 랜덤 뽑기    
    vehicleVelocitys = []
    
    for i in range(vehicleMAX):
        vehicleVelocitys.append(random.random() * 100)
        
        
    Vehicles = []
    
    for i in range(vehicleMAX): 
        Vehicles.append(Vehicle(position=(vehiclePositions[i][0],vehiclePositions[i][1]), velocity = vehicleVelocitys[i], direction=DIRECTION.UP, map_width=len(MAP[0]), map_height=len(MAP)))
    
    return Vehicles  
 
    
        
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

def ClearVehiclefromMAP(MAP):
    for row in range(len(MAP)):
        for col in range(len(MAP[row])):
            if MAP[row][col] == 2:
                MAP[row][col] = 0
    return MAP
            
#시물레이션 맵 가로
MAP_WIDTH = 50
#시뮬레이션 맵 세로 크기

MAP_HEIGHT = 50

#현재 Map에 차량이 있을 수 
VEHICLE_MAX = 5

#BaseStation Position
MM_BASESTATION_POSITION = [25,25]

#Beam의 방사각
MM_BASESTATION_BEAM_WDITH = 30

#설치한 Beam의 갯수
MM_BASESTATION_BEAM = int(360 / MM_BASESTATION_BEAM_WDITH)

#한번에 선택가능한 최대 beam의 갯수 m
SELECT_MAX = 3

#도로 설정 가로축 길 1개, 세로축 길 1개
TRAFFIC_ROADS = [[(0,35),(MAP_HEIGHT, 35)], [(35,0),(35, MAP_WIDTH)], 
                 [(15,0),(15, MAP_WIDTH)], [(0,10),(MAP_HEIGHT, 10)]]

EXPOLITATION_CONTOL = 10
MAP = CreateMAP(width=MAP_WIDTH, height=MAP_HEIGHT,basestationposition=MM_BASESTATION_POSITION,roads=TRAFFIC_ROADS)

   
contextDataList = []
contextDataList.append(contextPartitialData(CONTEXTDATA_NAME.POSITIONY, MAP_HEIGHT))
contextDataList.append(contextPartitialData(CONTEXTDATA_NAME.POSITIONX, MAP_WIDTH))

mmBS = mmBaseStation(MAP=MAP, 
                     position=MM_BASESTATION_POSITION, 
                     beamWidth = MM_BASESTATION_BEAM_WDITH, 
                     beamCount = MM_BASESTATION_BEAM, 
                     contextDataList = contextDataList,
                     selectMax = SELECT_MAX, 
                     expolitationControl = EXPOLITATION_CONTOL )

pixelPerHeight = 1 / MAP_HEIGHT
pixelPerWidth = 1 / MAP_WIDTH

Episode = 10
step = 10

for epi in range(Episode):
    
    for ste in range(step):
        
        #지도에서 차량 지우기
        MAP = ClearVehiclefromMAP(MAP)
        
        #차량 랜덤 생성
        Vehicles = CreateRandomVehicle(MAP, VEHICLE_MAX)   
        
        mmBS.MAP = MAP
        
        mmBS.Action(Vehicles)
        
        #랜덤 차량 위치 표시
        for vehicle in Vehicles:
            MAP[vehicle.position[0],vehicle.position[1]]= 2
        
        plt.imshow(MAP, interpolation='nearest', cmap=plt.cm.bone_r)






