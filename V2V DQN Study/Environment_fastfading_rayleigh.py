# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 15:53:27 2021

@author: CNL-B3
Fast fading : Reighy
"""

from __future__ import division
import numpy as np
import time
import random
import math
import matplotlib.pylab as plt
import matplotlib.patches as patches


MAX_SNR_DB = 30 #releigh

up_lanes = [3.5/2,3.5/2 + 3.5,250+3.5/2, 250+3.5+3.5/2, 500+3.5/2, 500+3.5+3.5/2]
down_lanes = [250-3.5-3.5/2,250-3.5/2,500-3.5-3.5/2,500-3.5/2,750-3.5-3.5/2,750-3.5/2]
left_lanes = [3.5/2,3.5/2 + 3.5,433+3.5/2, 433+3.5+3.5/2, 866+3.5/2, 866+3.5+3.5/2]
right_lanes = [433-3.5-3.5/2,433-3.5/2,866-3.5-3.5/2,866-3.5/2,1299-3.5-3.5/2,1299-3.5/2]

width = 750
height = 1299
n_Veh = 20

v2vlink_fastfading_normal = []
v2ilink_fastfading_normal = []
v2vlink_fastfading_rayleight = []
v2ilink_fastfading_rayleight = []


# This file is revised for more precise and concise expression.
"""
분석 순서

1. 도로 환경 설정 찾기. (완료, 그래프로 뛰움)   - add_new_vehicles_by_number 함수
2. 차량의 Dropping model 찾기. (완료, 그래프 표시중) - add_new_vehicles_by_number 함
3. V2V, V2I Shadowing model 찾기. (완료) - add_new_vehicles_by_number 함수
4. 차량의 위치와 방향 update 찾기.(완료) - renew_positions 함수

5. Large fading update 찾기. (완료) - update_large_fading 함수
   - V2VChannels class의 update_positions 함수 (분석 완료)
   - V2IChannels class의 update_positions 함수 (분석 완료)
   
   - V2VChannels class의 update_pathloss 함수 (분석 완료)
   - V2IChannels class의 update_pathloss 함수 (분석 완료)
   
   - V2VChannels class의 update_shadow 함수 (분석 완료)
   - V2IChannels class의 update_shadow 함수 (분석 완료)
   
6. Small fading update 찾기. (완료) - update_Small_fading 함수
   - V2VChannels class의 update_fast_fading 함수(거이 랜덤값임)
   - V2IChannels class의 update_fast_fading 함수(거이 랜덤값임)
   
   
5. 5G Channel 모델 찾기
"""

#SNRCreater 생성 Class     
class SNRCreater:
    #db값을 실수로
    def dB2real(self,fDBValue):
        return pow(10.0, fDBValue/10.0);

    #실수를 db값으로
    def	real2dB(self,fRealValue):
        return 10.0 * math.log10(fRealValue)
 
    #레일리 페이딩 기반 랜덤값 생성
    def GetReleighSNR(self,fAvgValue):
        value = random.random()  
        return self.real2dB(-self.dB2real(fAvgValue) * math.log(1.0 - value))
    
    def GetRageRandom(self, fMaxValue, fMinValue):
        return random.uniform(fMinValue,fMaxValue)

snrCreater = SNRCreater()
    
class V2Vchannels:              
    # Simulator of the V2V Channels
    def __init__(self, n_Veh, n_RB):
        self.t = 0
        self.h_bs = 1.5 #기지국의 안테나 높이 m
        self.h_ms = 1.5 #차량의 안테나 높이 m
        self.fc = 2 #캐리어 주파수 2GHz
        self.decorrelation_distance = 10 #decorrelation_distance 10m
        self.shadow_std = 3 #shadowing 표준편차 3, 평균은 0임
        self.n_Veh = n_Veh # 차량의 갯수
        self.n_RB = n_RB # resource 블록 -> 차량들이 점유할 수 있는 주파수블럭의 수
        self.update_shadow([])
        self.vehicles = []
    
    def set_vehicles(self, vehicles : list):
        self.vehicles = vehicles
        
    def update_positions(self, positions):
        self.positions = positions
        
    def update_pathloss(self):
        self.PathLoss = np.zeros(shape=(len(self.positions),len(self.positions)))
        for i in range(len(self.positions)):
            for j in range(len(self.positions)):
                self.PathLoss[i][j] = self.get_path_loss(self.positions[i], self.positions[j])
                
    def update_shadow(self, delta_distance_list):
        """
        

        Parameters
        ----------
        delta_distance_list : TYPE
        

        Returns
        -------
        None.

        """
        delta_distance = np.zeros((len(delta_distance_list), len(delta_distance_list)))
        
        for i in range(len(delta_distance)):
            for j in range(len(delta_distance)):
                delta_distance[i][j] = delta_distance_list[i] + delta_distance_list[j]
        
        #처음에는 가우시안 분포로 초기화
        if len(delta_distance_list) == 0: 
            self.Shadow = np.random.lognormal(0,self.shadow_std, size=(self.n_Veh, self.n_Veh))
        else:#이후 업데이트는 다음 공식따름
            self.Shadow = np.exp(-1*(delta_distance/self.decorrelation_distance)) * self.Shadow +\
                         np.sqrt(1 - np.exp(-2*(delta_distance/self.decorrelation_distance))) * np.random.normal(0, self.shadow_std, size = (self.n_Veh, self.n_Veh))
                         
    def update_fast_fading(self):
        
        """
        [
         [ [r,r,r] , [r,r,r], [r,r,r] ],
         [ [r,r,r] , [r,r,r], [r,r,r] ],
         [ [r,r,r] , [r,r,r], [r,r,r] ],
                  ]
        """
        h = 1/np.sqrt(2) * (np.random.normal(size=(self.n_Veh, self.n_Veh, self.n_RB) )  + 1j * np.random.normal(size=(self.n_Veh, self.n_Veh, self.n_RB)))
        self.FastFading = 20 * np.log10(np.abs(h)) #magnitude to db 20을 곱하는 이유 -> Volt -> SNR dbm으로 변환을 위함. -> 레일리 페이딩 써도 될듯..? 차량들의 평균 SNR을 랜덤으로 설정하고 지정
        v2vlink_fastfading_normal.append(self.FastFading[0][1][0])
        
    def update_fast_fading_reyleigh(self):    
        """
        rayleight fading 함수 추가
        """
        self.FastFading_rayleight = []
        for vehicle_i in range(len(self.vehicles)):
            vehicles_i_to_j_snrs_blocks = []
            for vehicle_j in range(len(self.vehicles)):
                resourceBlocks = []
                for resourceIndex in range(self.n_RB): 
                    fillValue = snrCreater.GetReleighSNR(self.vehicles[vehicle_j].snrAverage_dB)
                    
                    if vehicle_i == vehicle_j:                  
                        fillValue = 0
                        
                    resourceBlocks.append(fillValue)
                    
                vehicles_i_to_j_snrs_blocks.append(resourceBlocks)
        
            self.FastFading_rayleight.append(vehicles_i_to_j_snrs_blocks)
        
        v2vlink_fastfading_rayleight.append(self.FastFading_rayleight[0][1][0])
        
    def get_path_loss(self, position_A, position_B):
        #상호 차량간의 거리 계산 d1 : x 좌표, d2 : y 좌
        """
        WINNER 2 채널모델의 4.3 pathloss, B1 시나리오를 따름.
        fc : 캐리어 주파수
        h_bs : 송신 vehicle 의 안테나 높이
        h_ms : 수신 vehicle 의 안테나 높이
        d1 : 두차량의 x축 거리
        d2 : 두차량의 y축 거리
        d : d1 과 d2의 대각거리
        d_bp : 4 * (self.h_bs - 1) * (self.h_ms - 1) * self.fc * (10**9)/(3*10**8) 
        
        Path loss 계산 방법
        
        
        LOS 경우 
        두차량의 x, y축 거리 값이 둘중 하나라도 7m 미만
        그외
        NLOS 계산
        """
        d1 = abs(position_A[0] - position_B[0])
        d2 = abs(position_A[1] - position_B[1])
        
        #대각 거리 계산
        d = math.hypot(d1,d2)+0.001
        
        
        d_bp = 4 * (self.h_bs - 1) * (self.h_ms - 1) * self.fc * (10**9)/(3*10**8)
        
        #WINNER2 채널모델 기반 pathloss 계산함.
        
        def PL_Los(d):
            if d <= 3:#두차량의 거리가 3m 이하이면 A = 22.7, B=41.0, C=20, 거리는 3m로 고정 계산
                return 22.7 * np.log10(3) + 41 + 20*np.log10(self.fc/5)
            else:#두차량의 거리가 3m 이상이면 A = 22.7, B=41.0, C=20으로 계산
                if d < d_bp: #두차량의 거리가 d_bp보다 작으면 위의 공식에서 거리는 실측으로 계산
                    return 22.7 * np.log10(d) + 41 + 20 * np.log10(self.fc/5)
                else:#두차량의 거리가 d_bp보다 크면 
                    return 40.0 * np.log10(d) + 9.45 - 17.3 * np.log10(self.h_bs) - 17.3 * np.log10(self.h_ms) + 2.7 * np.log10(self.fc/5)
        def PL_NLos(d_a,d_b):
                n_j = max(2.8 - 0.0024*d_b, 1.84)
                return PL_Los(d_a) + 20 - 12.5*n_j + 10 * n_j * np.log10(d_b) + 3*np.log10(self.fc/5)
            
        if min(d1,d2) < 7:  # (x,y 좌표 둘중 1개라도 7이하 이면 Los로 path loss 계산)
            PL = PL_Los(d)
            self.ifLOS = True
            self.shadow_std = 3 
        else: # (그외 NLos로 path loss 계산)
            PL = min(PL_NLos(d1,d2), PL_NLos(d2,d1))
            self.ifLOS = False
            self.shadow_std = 4                      # if Non line of sight, the std is 4
        return PL

class V2Ichannels: 
    # Simulator of the V2I channels
    def __init__(self, n_Veh, n_RB):
        self.h_bs = 25 #기지국의 높이
        self.h_ms = 1.5 #안테나의 높이
        self.Decorrelation_distance = 50        
        self.BS_position = [750/2, 1299/2]    # 베이스스테이션은 정가운대에 있음
        self.shadow_std = 8
        self.n_Veh = n_Veh
        self.n_RB = n_RB
        self.update_shadow([])
        self.vehicles = []
        
    def update_positions(self, positions):
        self.positions = positions
    
    def set_vehicles(self, vehicles : list):
        self.vehicles = vehicles
        
    def update_pathloss(self):
        self.PathLoss = np.zeros(len(self.positions))
        for i in range(len(self.positions)):
            d1 = abs(self.positions[i][0] - self.BS_position[0])
            d2 = abs(self.positions[i][1] - self.BS_position[1])
            distance = math.hypot(d1,d2) #대각거리 계산 # change from meters to kilometers
            self.PathLoss[i] = 128.1 + 37.6*np.log10( math.sqrt(distance**2 + (self.h_bs-self.h_ms)**2) / 1000 )

    def update_shadow(self, delta_distance_list):
        if len(delta_distance_list) == 0:  # initialization
            self.Shadow = np.random.normal(0, self.shadow_std, self.n_Veh)
        else: 
            delta_distance = np.asarray(delta_distance_list)
            self.Shadow = np.exp(-1*(delta_distance/self.Decorrelation_distance))* self.Shadow +\
                          np.sqrt(1-np.exp(-2*(delta_distance/self.Decorrelation_distance)))*np.random.normal(0,self.shadow_std, self.n_Veh)
    def update_fast_fading(self):
        h = 1/np.sqrt(2) * (np.random.normal(size = (self.n_Veh, self.n_RB)) + 1j* np.random.normal(size = (self.n_Veh, self.n_RB)))
        self.FastFading = 20 * np.log10(np.abs(h))
        v2ilink_fastfading_normal.append(self.FastFading[0][1])
         
    def update_fast_fading_reyleigh(self):    
        """
        rayleight fading 함수 추가
        """
        self.FastFading_rayleight = []
        for vehicle_index in range(len(self.vehicles)):
            resourceBlocks = []
            for resourceIndex in range(self.n_RB):             
                resourceBlocks.append(snrCreater.GetReleighSNR(self.vehicles[vehicle_index].snrAverage_dB))
            self.FastFading_rayleight.append(resourceBlocks)
        
        v2ilink_fastfading_rayleight.append(self.FastFading_rayleight[0][1])
            

class Vehicle:
    # Vehicle simulator: include all the information for a vehicle
    def __init__(self, start_position, start_direction, velocity):
        self.position = start_position
        self.direction = start_direction
        self.velocity = velocity
        self.snrAverage_dB = 0 #random.uniform(1.0, 10.0) #레일레이 페이딩을 위함
        self.neighbors = []
        self.destinations = []
        
class Environ:
    # Enviroment Simulator: Provide states and rewards to agents. 
    # Evolve to new state based on the actions taken by the vehicles.
    
    """
    TvT 논문의 Simulation 환경을 제공
    state와 action에 대한 reward를 agent에게 전달
    vehicles이 취한 행동에 따라 새로운 상태로 전환함.
    """

    def __init__ (self, down_lane, up_lane, left_lane, right_lane, width, height, n_Veh):
        self.timestep = 0.01 # 시간은 0.01초 단위 
        self.down_lanes = down_lane   # 아래 도로 ? 
        self.up_lanes = up_lane       # 위 도로 ?
        self.left_lanes = left_lane   # 왼쪽 도로 ?
        self.right_lanes = right_lane # 오른쪽 도로 ?
        
        self.width = width           # 지도의 가로 방향
        self.height = height         # 지도의 세로 방향
        
        self.vehicles = []           # 차량의 수
        self.demands = []            # 요구하는 차량의 수 ?
        
        self.V2V_power_dB = 23       # v2v link의 dBm 
        self.V2I_power_dB = 23       # v2i link의 dBm
        self.V2V_power_dB_List = [23, 10, 5]            # v2v link의 종류별 파워 레벨
        #self.V2V_power = 10**(self.V2V_power_dB)
        #self.V2I_power = 10**(self.V2I_power_dB)
        self.sig2_dB = -114          #노이즈 파워 dbm 단위
        self.bsAntGain = 8           #기지국 안테나 gain
        self.bsNoiseFigure = 5       #기지국 수신 잡음 지수
        self.vehAntGain = 3          #차량 안테나 gain
        self.vehNoiseFigure = 9      #차량 수신 잡음 지수
        self.sig2 = 10**(self.sig2_dB/10) #노이즈 파워 watt 단위
        self.V2V_Shadowing = []     #v2v link의 Shadowing : 안쓰임 V2V Channel class에서 모두 다룸
        self.V2I_Shadowing = []     #v2i link의 Shadowing : 안쓰임 V2I Channel class에서 모두 다룸
        self.delta_distance = []    #한스탭 사이에서 차량이동 거리 
        self.n_RB = 20              #resource block의 수 차량들이 주파수 점유 할 수 있는 수
        self.n_Veh = n_Veh            #현재 시뮬레이션에 차량의 
        
        self.V2Vchannels = V2Vchannels(self.n_Veh, self.n_RB)  # V2V 채널 Class, 차량 수와 동일함.
        self.V2Ichannels = V2Ichannels(self.n_Veh, self.n_RB)

        self.V2V_Interference_all = np.zeros((self.n_Veh, 3, self.n_RB)) + self.sig2
        self.n_step = 0
        
    def add_new_vehicles(self, start_position, start_direction, start_velocity):    
        self.vehicles.append(Vehicle(start_position, start_direction, start_velocity))
        self.V2Vchannels.set_vehicles(self.vehicles)
        self.V2Ichannels.set_vehicles(self.vehicles)
        
        
    """
    
    입력 파라미터 n * 4 씩 차량 추가 
    시뮬레이션의 lanes list 값
    위쪽   방향의 도로  up_lanes    = [3.5/2,           3.5/2 + 3.5,   250+3.5/2,       250+3.5+3.5/2,   500+3.5/2,       500+3.5+3.5/2] 6개의 방향 차선이 존재함
    아래쪽 방향의 도로  down_lanes  = [250-3.5-3.5/2,   250-3.5/2,     500-3.5-3.5/2,   500-3.5/2,       750-3.5-3.5/2,   750-3.5/2]     6개의 방향 차선이 존재함
    왼쪽   방향의 도로  left_lanes  = [3.5/2,           3.5/2 + 3.5,   433+3.5/2,       433+3.5+3.5/2,   866+3.5/2,       866+3.5+3.5/2] 6개의 방향 차선이 존재함
    오른쪽 방향의 도로  right_lanes = [433-3.5-3.5/2,   433-3.5/2,     866-3.5-3.5/2,   866-3.5/2,       1299-3.5-3.5/2,  1299-3.5/2]    6개의 방향 차선이 존재함
    
    n : 1 ~ (n-1) 씩 증가
    차량 1 * n 번째: 아래쪽 방향, x : down_lanes(list)의 값중에 한곳에서 랜덤으로 출발 도로를 결정 함,  y : 0 ~ 1299의 값중에 한곳에서 랜덤으로 출발
    차량 2 * n 번째: 위쪽 방향,   x : up_lanes(list)의 값중에 한곳에서 랜덤으로 출발 도로를 결정 함,    y : 0 ~ 1299의 값중에 한곳에서 랜덤으로 출발
    차량 3 * n 번째: 왼쪽 방향,   x : 0 ~ 750 의 값중에 한곳에서 랜덤으로 출발,                        y : left_lanes(linst)의 값중에 한곳에서 랜덤으로 출발 도로를 결정함
    차량 4 * n 번째: 오른쪽 방향, x : 0 ~ 750의 값중에 한곳에서 랜덤으로 출발,                         y : right_lanes(linst)의 값중에 한곳에서 랜덤으로 출발 도로를 결정함
    
    """
    def add_new_vehicles_by_number(self, n):
        
        for i in range(n):
            
            """
            ind : 정수형 (0 ~ down lane 사이즈)사이의 랜덤 값 추출 
            width = 750 : 지도의 가로 사이즈가 750
            height = 1299 : 지도의 세로 사이즈가 1299
            1. 차량의 속력 : random.randint(10,15) 10이상 15이하 정수 랜덤값 생성    
            2. 차량의 방향 : 하, 상, 왼, 오 순으로 생성
            3. 차량의 위치 : 각 방향별로 6개의 도로가 있으며, 6개중 1개 랜덤(1~6)선택 후 선택된 도로의 좌표 안에서 랜덤(수평 방향 도로인 경우 width 0~749, 수직 방향 도로인 경우 height 0~1298) 선택
            4. V2V Shadowing : np.random.normal(0,3) 평균 0, 표준 편차 3인 랜덤값 생성
            5. V2I Shadowing : np.random.normal(0,8) 평균 0, 표준 편차 3인 랜덤값 생성
            6. delta_distance : 등록된 차량의 순서대로 velocity 저장
            """
            
            ind = np.random.randint(0,len(self.down_lanes))
            start_position = [self.down_lanes[ind], random.randint(0,self.height)]
            start_direction = 'd'
            self.add_new_vehicles(start_position,start_direction,random.randint(10,15))
            
            start_position = [self.up_lanes[ind], random.randint(0,self.height)]
            start_direction = 'u'
            self.add_new_vehicles(start_position,start_direction,random.randint(10,15))
            
            start_position = [random.randint(0,self.width), self.left_lanes[ind]]
            start_direction = 'l'
            self.add_new_vehicles(start_position,start_direction,random.randint(10,15))
            
            start_position = [random.randint(0,self.width), self.right_lanes[ind]]
            start_direction = 'r'
            self.add_new_vehicles(start_position,start_direction,random.randint(10,15))
        
        #평균 0, 표준 편차 3 인 V2V Link 생성 차량의 갯수가 N개 일때 NxN 매트릭스가 생성됨 ex 차량 수 5 -> 5x5
        self.V2V_Shadowing = np.random.normal(0, 3, [len(self.vehicles), len(self.vehicles)])
        
        #평균 0, 표준 편차 8 인 V2I Link 생성 차량의 갯수가 N개 일때 NxN 매트릭스가 생성됨 ex 차량 수 5 -> 5x5
        self.V2I_Shadowing = np.random.normal(0, 8, len(self.vehicles))
        
        #각 차량의 속력 저장
        self.delta_distance = ([c.velocity for c in self.vehicles])
        
        
        #self.renew_channel()
        """
        차량의 방향과 위치를 변화시키는 함수
        """
    def renew_positions(self):
        
        
        # ========================================================
        # This function update the position of each vehicle
        # ===========================================================
        i = 0
        #for i in range(len(self.position)):
        while(i < len(self.vehicles)):
            #print ('start iteration ', i)
            #print(self.position, len(self.position), self.direction)
            delta_distance = self.vehicles[i].velocity * self.timestep
            change_direction = False
            
            """
            up 방향으로 가는차량
            0.4 확률로 왼쪽으로 회전
            0.4 확률로 오른쪽으로 회전
            """
            if self.vehicles[i].direction == 'u':
                #print ('len of position', len(self.position), i)
                for j in range(len(self.left_lanes)):
                    
                    if (self.vehicles[i].position[1] <=self.left_lanes[j]) and ((self.vehicles[i].position[1] + delta_distance) >= self.left_lanes[j]):   # came to an cross
                        if (random.uniform(0,1) < 0.4):
                            self.vehicles[i].position = [self.vehicles[i].position[0] - (delta_distance - (self.left_lanes[j] - self.vehicles[i].position[1])), self.left_lanes[j] ] 
                            self.vehicles[i].direction = 'l'
                            change_direction = True
                            break
                if change_direction == False :
                    for j in range(len(self.right_lanes)):
                        if (self.vehicles[i].position[1] <=self.right_lanes[j]) and ((self.vehicles[i].position[1] + delta_distance) >= self.right_lanes[j]):
                            if (random.uniform(0,1) < 0.4):
                                self.vehicles[i].position = [self.vehicles[i].position[0] + (delta_distance + (self.right_lanes[j] - self.vehicles[i].position[1])), self.right_lanes[j] ] 
                                self.vehicles[i].direction = 'r'
                                change_direction = True
                                break
                if change_direction == False:
                    self.vehicles[i].position[1] += delta_distance
                    
            if (self.vehicles[i].direction == 'd') and (change_direction == False):
                #print ('len of position', len(self.position), i)
                for j in range(len(self.left_lanes)):
                    if (self.vehicles[i].position[1] >=self.left_lanes[j]) and ((self.vehicles[i].position[1] - delta_distance) <= self.left_lanes[j]):  # came to an cross
                        if (random.uniform(0,1) < 0.4):
                            self.vehicles[i].position = [self.vehicles[i].position[0] - (delta_distance - ( self.vehicles[i].position[1]- self.left_lanes[j])), self.left_lanes[j] ] 
                            #print ('down with left', self.vehicles[i].position)
                            self.vehicles[i].direction = 'l'
                            change_direction = True
                            break
                if change_direction == False :
                    for j in range(len(self.right_lanes)):
                        if (self.vehicles[i].position[1] >=self.right_lanes[j]) and (self.vehicles[i].position[1] - delta_distance <= self.right_lanes[j]):
                            if (random.uniform(0,1) < 0.4):
                                self.vehicles[i].position = [self.vehicles[i].position[0] + (delta_distance + ( self.vehicles[i].position[1]- self.right_lanes[j])),self.right_lanes[j] ] 
                                #print ('down with right', self.vehicles[i].position)
                                self.vehicles[i].direction = 'r'
                                change_direction = True
                                break
                if change_direction == False:
                    self.vehicles[i].position[1] -= delta_distance
            if (self.vehicles[i].direction == 'r') and (change_direction == False):
                #print ('len of position', len(self.position), i)
                for j in range(len(self.up_lanes)):
                    if (self.vehicles[i].position[0] <= self.up_lanes[j]) and ((self.vehicles[i].position[0] + delta_distance) >= self.up_lanes[j]):   # came to an cross
                        if (random.uniform(0,1) < 0.4):
                            self.vehicles[i].position = [self.up_lanes[j], self.vehicles[i].position[1] + (delta_distance - (self.up_lanes[j] - self.vehicles[i].position[0]))]
                            change_direction = True
                            self.vehicles[i].direction = 'u'
                            break
                if change_direction == False :
                    for j in range(len(self.down_lanes)):
                        if (self.vehicles[i].position[0] <= self.down_lanes[j]) and ((self.vehicles[i].position[0] + delta_distance) >= self.down_lanes[j]):
                            if (random.uniform(0,1) < 0.4):
                                self.vehicles[i].position = [self.down_lanes[j], self.vehicles[i].position[1] - (delta_distance - (self.down_lanes[j] - self.vehicles[i].position[0]))]
                                change_direction = True
                                self.vehicles[i].direction = 'd'
                                break
                if change_direction == False:
                    self.vehicles[i].position[0] += delta_distance
            if (self.vehicles[i].direction == 'l') and (change_direction == False):
                for j in range(len(self.up_lanes)):
                    
                    if (self.vehicles[i].position[0] >= self.up_lanes[j]) and ((self.vehicles[i].position[0] - delta_distance) <= self.up_lanes[j]):   # came to an cross
                        if (random.uniform(0,1) < 0.4):
                            self.vehicles[i].position = [self.up_lanes[j], self.vehicles[i].position[1] + (delta_distance - (self.vehicles[i].position[0] - self.up_lanes[j]))]
                            change_direction = True
                            self.vehicles[i].direction = 'u'
                            break
                if change_direction == False :
                    for j in range(len(self.down_lanes)):
                        if (self.vehicles[i].position[0] >= self.down_lanes[j]) and ((self.vehicles[i].position[0] - delta_distance) <= self.down_lanes[j]):
                            if (random.uniform(0,1) < 0.4):
                                self.vehicles[i].position = [self.down_lanes[j], self.vehicles[i].position[1] - (delta_distance - (self.vehicles[i].position[0] - self.down_lanes[j]))]
                                change_direction = True
                                self.vehicles[i].direction = 'd'
                                break
                    if change_direction == False:
                        self.vehicles[i].position[0] -= delta_distance
            # if it comes to an exit
            if (self.vehicles[i].position[0] < 0) or (self.vehicles[i].position[1] < 0) or (self.vehicles[i].position[0] > self.width) or (self.vehicles[i].position[1] > self.height):
            # delete
            #    print ('delete ', self.position[i])
                if (self.vehicles[i].direction == 'u'):
                    self.vehicles[i].direction = 'r'
                    self.vehicles[i].position = [self.vehicles[i].position[0], self.right_lanes[-1]]
                else:
                    if (self.vehicles[i].direction == 'd'):
                        self.vehicles[i].direction = 'l'
                        self.vehicles[i].position = [self.vehicles[i].position[0], self.left_lanes[0]]
                    else:
                        if (self.vehicles[i].direction == 'l'):
                            self.vehicles[i].direction = 'u'
                            self.vehicles[i].position = [self.up_lanes[0],self.vehicles[i].position[1]]
                        else:
                            if (self.vehicles[i].direction == 'r'):
                                self.vehicles[i].direction = 'd'
                                self.vehicles[i].position = [self.down_lanes[-1],self.vehicles[i].position[1]]
                
            i += 1
    def test_channel(self):
        # ===================================
        #   test the V2I and the V2V channel 
        # ===================================
        
        self.n_step = 0 #시뮬레이션 스탭
        self.vehicles = [] #차량 등록 list
        n_Veh = 20 #테스트 차량의수 20대
        self.n_Veh = n_Veh  
        self.add_new_vehicles_by_number(int(self.n_Veh/4)) #4를 나누는 이유 add_new_vehicles_by_number함수에서 1번 루프당 4대의 차를 생성하기 때문
       
        step = 1000 #시뮬레이션 스탭 총 1000스탭
        time_step = 0.1  # every 0.1s update, 100ms마다 한번씩 업데이트 예정
        
        for i in range(step):
            self.renew_positions()  #위치 재갱신
            positions = [c.position for c in self.vehicles]
            
            self.update_large_fading(positions, time_step) #차량의 위치를 보고 large fading을 재갱신함
            self.update_small_fading()#small fading은 fast fading에 의거해서 재갱신하며 차량의 갯수, 리소스 블럭의 수로 다룸.
            
            print("Time step: ", i)
            print(" ============== V2I ===========")
            print("Path Loss: ", self.V2Ichannels.PathLoss)
            print("Shadow:",  self.V2Ichannels.Shadow)
            print("Fast Fading: ",  self.V2Ichannels.FastFading)
            print(" ============== V2V ===========")
            print("Path Loss: ", self.V2Vchannels.PathLoss[0:3])
            print("Shadow:", self.V2Vchannels.Shadow[0:3])
            print("Fast Fading: ", self.V2Vchannels.FastFading[0:3])

    def update_large_fading(self, positions, time_step):
        
        """
        기존에 저장되어있는 V2I, V2V Channels 의 position 업데이트 
        position만 재갱신
        """
        self.V2Ichannels.update_positions(positions)
        self.V2Vchannels.update_positions(positions)
        
        """
        기존에 저장되어있는 V2I, V2V Channels 의 pathloss 업데이트
        position 변화에 따른 pathloss 계산
        """
        self.V2Ichannels.update_pathloss()
        self.V2Vchannels.update_pathloss()
        
        """
        기존에 저장되어있는 V2I, V2V Channels 의 shadow 업데이트, 속력도 필요함
        delta_distance, 100ms 지났을때의 이동거리.
        """
        delta_distance = time_step * np.asarray([c.velocity for c in self.vehicles])
        self.V2Ichannels.update_shadow(delta_distance)
        self.V2Vchannels.update_shadow(delta_distance)
        
    def update_small_fading(self):
        self.V2Ichannels.update_fast_fading()
        self.V2Vchannels.update_fast_fading()
        
        self.V2Ichannels.update_fast_fading_reyleigh()
        self.V2Vchannels.update_fast_fading_reyleigh()
   
    #아래 주석 사이의 함수들은 DQN적용을 위해 사용함, Interference 계산도 있음!
    
    def renew_neighbor(self):   
        # ==========================================
        # update the neighbors of each vehicle.
        # 상호 차량들의 거리를 보고 현재 차량의 vehicle class의 neighbor를 재갱신한다.
        # ===========================================
        for i in range(len(self.vehicles)):
            self.vehicles[i].neighbors = []
            self.vehicles[i].actions = []
            #print('action and neighbors delete', self.vehicles[i].actions, self.vehicles[i].neighbors)
        Distance = np.zeros((len(self.vehicles),len(self.vehicles)))
        z = np.array([[complex(c.position[0],c.position[1]) for c in self.vehicles]])
        Distance = abs(z.T-z)
        for i in range(len(self.vehicles)):       
            sort_idx = np.argsort(Distance[:,i])
            for j in range(3):
                self.vehicles[i].neighbors.append(sort_idx[j+1])                
            destination = np.random.choice(sort_idx[1:int(len(sort_idx)/5)],3, replace = False)
            self.vehicles[i].destinations = destination
            
    def renew_channel(self):
        # ===========================================================================
        # This function updates all the channels including V2V and V2I channels
        # =============================================================================
        positions = [c.position for c in self.vehicles]
        self.V2Ichannels.update_positions(positions)
        self.V2Vchannels.update_positions(positions)
        self.V2Ichannels.update_pathloss()
        self.V2Vchannels.update_pathloss()
        delta_distance = 0.002 * np.asarray([c.velocity for c in self.vehicles])    # time slot is 2 ms. 
        self.V2Ichannels.update_shadow(delta_distance)
        self.V2Vchannels.update_shadow(delta_distance)
        self.V2V_channels_abs = self.V2Vchannels.PathLoss + self.V2Vchannels.Shadow + 50 * np.identity(
            len(self.vehicles))
        self.V2I_channels_abs = self.V2Ichannels.PathLoss + self.V2Ichannels.Shadow

    def renew_channels_fastfading(self):   
        # =======================================================================
        # This function updates all the channels including V2V and V2I channels
        # =========================================================================
        self.renew_channel()
        self.V2Ichannels.update_fast_fading()
        self.V2Vchannels.update_fast_fading()
        V2V_channels_with_fastfading = np.repeat(self.V2V_channels_abs[:, :, np.newaxis], self.n_RB, axis=2)
        self.V2V_channels_with_fastfading = V2V_channels_with_fastfading - self.V2Vchannels.FastFading
        V2I_channels_with_fastfading = np.repeat(self.V2I_channels_abs[:, np.newaxis], self.n_RB, axis=1)
        self.V2I_channels_with_fastfading = V2I_channels_with_fastfading - self.V2Ichannels.FastFading
        #print("V2I channels", self.V2I_channels_with_fastfading)
        
    def Compute_Performance_Reward_fast_fading_with_power(self, actions_power):   # revising based on the fast fading part
        actions = actions_power.copy()[:,:,0]  # the channel_selection_part
        power_selection = actions_power.copy()[:,:,1]
        Rate = np.zeros(len(self.vehicles))
        Interference = np.zeros(self.n_RB)  # V2V signal interference to V2I links
        for i in range(len(self.vehicles)):
            for j in range(len(actions[i,:])):
                if not self.activate_links[i,j]:
                    continue
                #print('power selection,', power_selection[i,j])  
                Interference[actions[i][j]] += 10**((self.V2V_power_dB_List[power_selection[i,j]]  - self.V2I_channels_with_fastfading[i, actions[i,j]] + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure)/10)  # fast fading

        self.V2I_Interference = Interference + self.sig2
        V2V_Interference = np.zeros((len(self.vehicles), 3))
        V2V_Signal = np.zeros((len(self.vehicles), 3))
        
        # remove the effects of none active links
        #print('shapes', actions.shape, self.activate_links.shape)
        #print(not self.activate_links)
        actions[(np.logical_not(self.activate_links))] = -1
        #print('action are', actions)
        for i in range(self.n_RB):
            indexes = np.argwhere(actions == i)
            for j in range(len(indexes)):
                #receiver_j = self.vehicles[indexes[j,0]].neighbors[indexes[j,1]]
                receiver_j = self.vehicles[indexes[j,0]].destinations[indexes[j,1]]
                # compute the V2V signal links
                V2V_Signal[indexes[j, 0],indexes[j, 1]] = 10**((self.V2V_power_dB_List[power_selection[indexes[j, 0],indexes[j, 1]]] - self.V2V_channels_with_fastfading[indexes[j][0]][receiver_j][i] + 2*self.vehAntGain - self.vehNoiseFigure)/10) 
                #V2V_Signal[indexes[j, 0],indexes[j, 1]] = 10**((self.V2V_power_dB_List[0] - self.V2V_channels_with_fastfading[indexes[j][0]][receiver_j][i])/10) 
                if i < self.n_Veh:
                    V2V_Interference[indexes[j,0],indexes[j,1]] += 10**((self.V2I_power_dB - self.V2V_channels_with_fastfading[i][receiver_j][i]+ 2*self.vehAntGain - self.vehNoiseFigure )/10)  # V2I links interference to V2V links  
                for k in range(j+1, len(indexes)):                  # computer the peer V2V links
                    #receiver_k = self.vehicles[indexes[k][0]].neighbors[indexes[k][1]]
                    receiver_k = self.vehicles[indexes[k][0]].destinations[indexes[k][1]]
                    V2V_Interference[indexes[j,0],indexes[j,1]] += 10**((self.V2V_power_dB_List[power_selection[indexes[k,0],indexes[k,1]]] - self.V2V_channels_with_fastfading[indexes[k][0]][receiver_j][i]+ 2*self.vehAntGain - self.vehNoiseFigure)/10)
                    V2V_Interference[indexes[k,0],indexes[k,1]] += 10**((self.V2V_power_dB_List[power_selection[indexes[j,0],indexes[j,1]]] - self.V2V_channels_with_fastfading[indexes[j][0]][receiver_k][i]+ 2*self.vehAntGain - self.vehNoiseFigure)/10)               
       
        self.V2V_Interference = V2V_Interference + self.sig2
        V2V_Rate = np.zeros(self.activate_links.shape)
        V2V_Rate[self.activate_links] = np.log2(1 + np.divide(V2V_Signal[self.activate_links], self.V2V_Interference[self.activate_links]))

        #print("V2V Rate", V2V_Rate * self.update_time_test * 1500)
        #print ('V2V_Signal is ', np.log(np.mean(V2V_Signal[self.activate_links])))
        V2I_Signals = self.V2I_power_dB-self.V2I_channels_abs[0:min(self.n_RB,self.n_Veh)] + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure
        V2I_Rate = np.log2(1 + np.divide(10**(V2I_Signals/10), self.V2I_Interference[0:min(self.n_RB,self.n_Veh)]))


         # -- compute the latency constraits --
        self.demand -= V2V_Rate * self.update_time_test * 1500    # decrease the demand
        self.test_time_count -= self.update_time_test               # compute the time left for estimation
        self.individual_time_limit -= self.update_time_test         # compute the time left for individual V2V transmission
        self.individual_time_interval -= self.update_time_test      # compute the time interval left for next transmission

        # --- update the demand ---
        
        new_active = self.individual_time_interval <= 0
        self.activate_links[new_active] = True
        self.individual_time_interval[new_active] = np.random.exponential(0.02, self.individual_time_interval[new_active].shape ) + self.V2V_limit
        self.individual_time_limit[new_active] = self.V2V_limit
        self.demand[new_active] = self.demand_amount
        #print("demand is", self.demand)
        #print('mean rate of average V2V link is', np.mean(V2V_Rate[self.activate_links]))
        
        # -- update the statistics---
        early_finish = np.multiply(self.demand <= 0, self.activate_links)        
        unqulified = np.multiply(self.individual_time_limit <=0, self.activate_links)
        self.activate_links[np.add(early_finish, unqulified)] = False 
        #print('number of activate links is', np.sum(self.activate_links)) 
        self.success_transmission += np.sum(early_finish)
        self.failed_transmission += np.sum(unqulified)
        #if self.n_step % 1000 == 0 :
        #    self.success_transmission = 0
        #    self.failed_transmission = 0
        failed_percentage = self.failed_transmission/(self.failed_transmission + self.success_transmission + 0.0001)
        # print('Percentage of failed', np.sum(new_active), self.failed_transmission, self.failed_transmission + self.success_transmission , failed_percentage)    
        return V2I_Rate, failed_percentage #failed_percentage

        
    def Compute_Performance_Reward_fast_fading_with_power_asyn(self, actions_power):   # revising based on the fast fading part
        # ===================================================
        #  --------- Used for Testing -------
        # ===================================================
        actions = actions_power[:,:,0]  # the channel_selection_part
        power_selection = actions_power[:,:,1]
        Interference = np.zeros(self.n_RB)   # Calculate the interference from V2V to V2I
        for i in range(len(self.vehicles)):
            for j in range(len(actions[i,:])):
                if not self.activate_links[i,j]:
                    continue
                Interference[actions[i][j]] += 10**((self.V2V_power_dB_List[power_selection[i,j]] - \
                                                     self.V2I_channels_with_fastfading[i, actions[i,j]] + \
                                                     self.vehAntGain + self.bsAntGain - self.bsNoiseFigure)/10)
        self.V2I_Interference = Interference + self.sig2
        V2V_Interference = np.zeros((len(self.vehicles), 3))
        V2V_Signal = np.zeros((len(self.vehicles), 3))
        Interfence_times = np.zeros((len(self.vehicles), 3))
        actions[(np.logical_not(self.activate_links))] = -1
        for i in range(self.n_RB):
            indexes = np.argwhere(actions == i)
            for j in range(len(indexes)):
                #receiver_j = self.vehicles[indexes[j,0]].neighbors[indexes[j,1]]
                receiver_j = self.vehicles[indexes[j,0]].destinations[indexes[j,1]]
                V2V_Signal[indexes[j, 0],indexes[j, 1]] = 10**((self.V2V_power_dB_List[power_selection[indexes[j, 0],indexes[j, 1]]] -\
                self.V2V_channels_with_fastfading[indexes[j][0]][receiver_j][i] + 2*self.vehAntGain - self.vehNoiseFigure)/10)
                #V2V_Signal[indexes[j, 0],indexes[j, 1]] = 10**((self.V2V_power_dB_List[0] - self.V2V_channels_with_fastfading[indexes[j][0]][receiver_j][i])/10) 
                if i<self.n_Veh:
                    V2V_Interference[indexes[j,0],indexes[j,1]] += 10**((self.V2I_power_dB - \
                    self.V2V_channels_with_fastfading[i][receiver_j][i] + 2*self.vehAntGain - self.vehNoiseFigure )/10)  # V2I links interference to V2V links
                for k in range(j+1, len(indexes)):
                    receiver_k = self.vehicles[indexes[k][0]].destinations[indexes[k][1]]
                    V2V_Interference[indexes[j,0],indexes[j,1]] += 10**((self.V2V_power_dB_List[power_selection[indexes[k,0],indexes[k,1]]] -\
                    self.V2V_channels_with_fastfading[indexes[k][0]][receiver_j][i]+ 2*self.vehAntGain - self.vehNoiseFigure)/10)
                    V2V_Interference[indexes[k,0],indexes[k,1]] += 10**((self.V2V_power_dB_List[power_selection[indexes[j,0],indexes[j,1]]] - \
                    self.V2V_channels_with_fastfading[indexes[j][0]][receiver_k][i]+ 2*self.vehAntGain - self.vehNoiseFigure)/10)
                    Interfence_times[indexes[j,0],indexes[j,1]] += 1
                    Interfence_times[indexes[k,0],indexes[k,1]] += 1               

        self.V2V_Interference = V2V_Interference + self.sig2
        V2V_Rate = np.log2(1 + np.divide(V2V_Signal, self.V2V_Interference))
        V2I_Signals = self.V2I_power_dB-self.V2I_channels_abs[0:min(self.n_RB,self.n_Veh)] + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure
        V2I_Rate = np.log2(1 + np.divide(10**(V2I_Signals/10), self.V2I_Interference[0:min(self.n_RB,self.n_Veh)]))
        #print("V2I information", V2I_Signals, self.V2I_Interference, V2I_Rate)
        
        # -- compute the latency constraits --
        self.demand -= V2V_Rate * self.update_time_asyn * 1500    # decrease the demand
        self.test_time_count -= self.update_time_asyn               # compute the time left for estimation
        self.individual_time_limit -= self.update_time_asyn         # compute the time left for individual V2V transmission
        self.individual_time_interval -= self.update_time_asyn     # compute the time interval left for next transmission

        # --- update the demand ---
        new_active = self.individual_time_interval <= 0
        self.activate_links[new_active] = True
        self.individual_time_interval[new_active] = np.random.exponential(0.02, self.individual_time_interval[new_active].shape) + self.V2V_limit
        self.individual_time_limit[new_active] = self.V2V_limit
        self.demand[new_active] = self.demand_amount
        
        # -- update the statistics---
        early_finish = np.multiply(self.demand <= 0, self.activate_links)        
        unqulified = np.multiply(self.individual_time_limit <=0, self.activate_links)
        self.activate_links[np.add(early_finish, unqulified)] = False
        self.success_transmission += np.sum(early_finish)
        self.failed_transmission += np.sum(unqulified)
        fail_percent = self.failed_transmission/(self.failed_transmission + self.success_transmission + 0.0001)            
        return V2I_Rate, fail_percent

    def Compute_Performance_Reward_Batch(self, actions_power, idx):    # add the power dimension to the action selection
        # ==================================================
        # ------------- Used for Training ----------------
        # ==================================================
        actions = actions_power.copy()[:,:,0]           #
        power_selection = actions_power.copy()[:,:,1]   #
        V2V_Interference = np.zeros((len(self.vehicles), 3))
        V2V_Signal = np.zeros((len(self.vehicles), 3))
        Interfence_times = np.zeros((len(self.vehicles), 3))    #  3 neighbors
        #print(actions)
        origin_channel_selection = actions[idx[0], idx[1]]
        actions[idx[0], idx[1]] = 100  # something not relavant
        for i in range(self.n_RB):
            indexes = np.argwhere(actions == i)
            #print('index',indexes)
            for j in range(len(indexes)):
                #receiver_j = self.vehicles[indexes[j,0]].neighbors[indexes[j,1]]
                receiver_j = self.vehicles[indexes[j,0]].destinations[indexes[j,1]]
                V2V_Signal[indexes[j, 0],indexes[j, 1]] = 10**((self.V2V_power_dB_List[power_selection[indexes[j, 0],indexes[j, 1]]] -\
                self.V2V_channels_with_fastfading[indexes[j,0], receiver_j, i]+ 2*self.vehAntGain - self.vehNoiseFigure)/10) 
                V2V_Interference[indexes[j,0],indexes[j,1]] +=  10**((self.V2I_power_dB- self.V2V_channels_with_fastfading[i,receiver_j,i] + \
                2*self.vehAntGain - self.vehNoiseFigure)/10)  # interference from the V2I links
                
                for k in range(j+1, len(indexes)):
                    receiver_k = self.vehicles[indexes[k,0]].destinations[indexes[k,1]]
                    V2V_Interference[indexes[j,0],indexes[j,1]] += 10**((self.V2V_power_dB_List[power_selection[indexes[k,0],indexes[k,1]]] - \
                    self.V2V_channels_with_fastfading[indexes[k,0],receiver_j,i] + 2*self.vehAntGain - self.vehNoiseFigure)/10)
                    V2V_Interference[indexes[k,0],indexes[k,1]] += 10**((self.V2V_power_dB_List[power_selection[indexes[j,0],indexes[j,1]]] - \
                    self.V2V_channels_with_fastfading[indexes[j,0], receiver_k, i] + 2*self.vehAntGain - self.vehNoiseFigure)/10)
                    Interfence_times[indexes[j,0],indexes[j,1]] += 1
                    Interfence_times[indexes[k,0],indexes[k,1]] += 1
                    
        self.V2V_Interference = V2V_Interference + self.sig2
        V2V_Rate_list = np.zeros((self.n_RB, len(self.V2V_power_dB_List)))  # the number of RB times the power level
        Deficit_list = np.zeros((self.n_RB, len(self.V2V_power_dB_List)))
        for i in range(self.n_RB):
            indexes = np.argwhere(actions == i)
            V2V_Signal_temp = V2V_Signal.copy()            
            #receiver_k = self.vehicles[idx[0]].neighbors[idx[1]]
            receiver_k = self.vehicles[idx[0]].destinations[idx[1]]
            for power_idx in range(len(self.V2V_power_dB_List)):
                V2V_Interference_temp = V2V_Interference.copy()
                V2V_Signal_temp[idx[0],idx[1]] = 10**((self.V2V_power_dB_List[power_idx] - \
                self.V2V_channels_with_fastfading[idx[0], self.vehicles[idx[0]].destinations[idx[1]],i] + 2*self.vehAntGain - self.vehNoiseFigure )/10)
                V2V_Interference_temp[idx[0],idx[1]] +=  10**((self.V2I_power_dB - \
                self.V2V_channels_with_fastfading[i,self.vehicles[idx[0]].destinations[idx[1]],i] + 2*self.vehAntGain - self.vehNoiseFigure)/10)
                for j in range(len(indexes)):
                    receiver_j = self.vehicles[indexes[j,0]].destinations[indexes[j,1]]
                    V2V_Interference_temp[idx[0],idx[1]] += 10**((self.V2V_power_dB_List[power_selection[indexes[j,0], indexes[j,1]]] -\
                    self.V2V_channels_with_fastfading[indexes[j,0],receiver_k, i] + 2*self.vehAntGain - self.vehNoiseFigure)/10)
                    V2V_Interference_temp[indexes[j,0],indexes[j,1]] += 10**((self.V2V_power_dB_List[power_idx]-\
                    self.V2V_channels_with_fastfading[idx[0],receiver_j, i] + 2*self.vehAntGain - self.vehNoiseFigure)/10)
                V2V_Rate_cur = np.log2(1 + np.divide(V2V_Signal_temp, V2V_Interference_temp))
                if (origin_channel_selection == i) and (power_selection[idx[0], idx[1]] == power_idx):
                    V2V_Rate = V2V_Rate_cur.copy()
                V2V_Rate_list[i, power_idx] = np.sum(V2V_Rate_cur)
                Deficit_list[i,power_idx] = 0 - 1 * np.sum(np.maximum(np.zeros(V2V_Signal_temp.shape), (self.demand - self.individual_time_limit * V2V_Rate_cur * 1500)))
        Interference = np.zeros(self.n_RB)  
        V2I_Rate_list = np.zeros((self.n_RB,len(self.V2V_power_dB_List)))    # 3 of power level
        for i in range(len(self.vehicles)):
            for j in range(len(actions[i,:])):
                if (i ==idx[0] and j == idx[1]):
                    continue
                Interference[actions[i][j]] += 10**((self.V2V_power_dB_List[power_selection[i,j]] - \
                self.V2I_channels_with_fastfading[i, actions[i][j]] + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure)/10) 
        V2I_Interference = Interference + self.sig2
        for i in range(self.n_RB):            
            for j in range(len(self.V2V_power_dB_List)):
                V2I_Interference_temp = V2I_Interference.copy()
                V2I_Interference_temp[i] += 10**((self.V2V_power_dB_List[j] - self.V2I_channels_with_fastfading[idx[0], i] + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure)/10)
                V2I_Rate_list[i, j] = np.sum(np.log2(1 + np.divide(10**((self.V2I_power_dB + self.vehAntGain + self.bsAntGain \
                - self.bsNoiseFigure-self.V2I_channels_abs[0:min(self.n_RB,self.n_Veh)])/10), V2I_Interference_temp[0:min(self.n_RB,self.n_Veh)])))
                     
        self.demand -= V2V_Rate * self.update_time_train * 1500
        self.test_time_count -= self.update_time_train
        self.individual_time_limit -= self.update_time_train
        self.individual_time_limit [np.add(self.individual_time_limit <= 0,  self.demand < 0)] = self.V2V_limit
        self.demand[self.demand < 0] = self.demand_amount
        if self.test_time_count == 0:
            self.test_time_count = 10
        return V2I_Rate_list, Deficit_list, self.individual_time_limit[idx[0], idx[1]]

    def Compute_Interference(self, actions):
        # ====================================================
        # Compute the Interference to each channel_selection
        # ====================================================
        V2V_Interference = np.zeros((len(self.vehicles), 3, self.n_RB)) + self.sig2
        if len(actions.shape) == 3:
            channel_selection = actions.copy()[:,:,0]
            power_selection = actions[:,:,1]
            channel_selection[np.logical_not(self.activate_links)] = -1
            for i in range(self.n_RB):
                for k in range(len(self.vehicles)):
                    for m in range(len(channel_selection[k,:])):
                        V2V_Interference[k, m, i] += 10 ** ((self.V2I_power_dB - self.V2V_channels_with_fastfading[i][self.vehicles[k].destinations[m]][i] + \
                        2 * self.vehAntGain - self.vehNoiseFigure)/10)
            for i in range(len(self.vehicles)):
                for j in range(len(channel_selection[i,:])):
                    for k in range(len(self.vehicles)):
                        for m in range(len(channel_selection[k,:])):
                            if (i==k) or (channel_selection[i,j] >= 0):
                                continue
                            V2V_Interference[k, m, channel_selection[i,j]] += 10**((self.V2V_power_dB_List[power_selection[i,j]] -\
                            self.V2V_channels_with_fastfading[i][self.vehicles[k].destinations[m]][channel_selection[i,j]] + 2*self.vehAntGain - self.vehNoiseFigure)/10)

        self.V2V_Interference_all = 10 * np.log10(V2V_Interference)
                
        
    def renew_demand(self):
        # generate a new demand of a V2V
        self.demand = self.demand_amount*np.ones((self.n_RB,3))
        self.time_limit = 10
        
    def act_for_training(self, actions, idx):
        # =============================================
        # This function gives rewards for training
        # ===========================================
        rewards_list = np.zeros(self.n_RB)
        action_temp = actions.copy()
        self.activate_links = np.ones((self.n_Veh,3), dtype = 'bool')
        V2I_rewardlist, V2V_rewardlist, time_left = self.Compute_Performance_Reward_Batch(action_temp,idx)
        self.renew_positions()
        self.renew_channels_fastfading()
        self.Compute_Interference(actions) 
        rewards_list = rewards_list.T.reshape([-1])
        V2I_rewardlist = V2I_rewardlist.T.reshape([-1])
        V2V_rewardlist = V2V_rewardlist.T.reshape([-1])
        V2I_reward = (V2I_rewardlist[actions[idx[0],idx[1], 0]+ 20*actions[idx[0],idx[1], 1]] -\
                      np.min(V2I_rewardlist))/(np.max(V2I_rewardlist) -np.min(V2I_rewardlist) + 0.000001)
        V2V_reward = (V2V_rewardlist[actions[idx[0],idx[1], 0]+ 20*actions[idx[0],idx[1], 1]] -\
                     np.min(V2V_rewardlist))/(np.max(V2V_rewardlist) -np.min(V2V_rewardlist) + 0.000001)
        lambdda = 0.1
        #print ("Reward", V2I_reward, V2V_reward, time_left)
        t = lambdda * V2I_reward + (1-lambdda) * V2V_reward
        #print("time left", time_left)
        #return t
        return t - (self.V2V_limit - time_left)/self.V2V_limit
        
    def act_asyn(self, actions):
        self.n_step += 1
        if self.n_step % 10 == 0:
            self.renew_positions()            
            self.renew_channels_fastfading()
        reward = self.Compute_Performance_Reward_fast_fading_with_power_asyn(actions)
        self.Compute_Interference(actions)
        return reward
    
    def act(self, actions):
        # simulate the next state after the action is given
        self.n_step += 1        
        reward = self.Compute_Performance_Reward_fast_fading_with_power(actions)
        self.renew_positions()            
        self.renew_channels_fastfading()
        self.Compute_Interference(actions)
        return reward
        
    def new_random_game(self, n_Veh = 0):
        # make a new game
        self.n_step = 0
        self.vehicles = []
        if n_Veh > 0:
            self.n_Veh = n_Veh
        self.add_new_vehicles_by_number(int(self.n_Veh/4))
        self.V2Vchannels = V2Vchannels(self.n_Veh, self.n_RB)  # number of vehicles
        self.V2Ichannels = V2Ichannels(self.n_Veh, self.n_RB)
        self.renew_channels_fastfading()
        self.renew_neighbor()
        self.demand_amount = 30
        self.demand = self.demand_amount * np.ones((self.n_Veh,3))
        self.test_time_count = 10
        self.V2V_limit = 0.1  # 100 ms V2V toleratable latency
        self.individual_time_limit = self.V2V_limit * np.ones((self.n_Veh,3))
        self.individual_time_interval = np.random.exponential(0.05, (self.n_Veh,3))
        self.UnsuccessfulLink = np.zeros((self.n_Veh,3))
        self.success_transmission = 0
        self.failed_transmission = 0
        self.update_time_train = 0.01  # 10ms update time for the training
        self.update_time_test = 0.002 # 2ms update time for testing
        self.update_time_asyn = 0.0002 # 0.2 ms update one subset of the vehicles; for each vehicle, the update time is 2 ms
        self.activate_links = np.zeros((self.n_Veh,3), dtype='bool')

    
#도로 생성 함수 (진입 못하는 도로 : -1, 도로 : 0,  BasetStation : 1, 차량 : 2)
def CreateMAP(width, height, roads):
    MAP = []
    
    #지도 생성
    for i in range(height):
        temp = []
        for j in range(width):
            temp.append(0)
        
        MAP.append(temp)
    
    #도로 생성
    for road in roads: 
             
        roadStartPositionX = int(road[0][0])
        roadStartPositionY = int(road[0][1])
        
        roadEndPositionX = int(road[1][0])
        roadEndPositionY = int(road[1][1])
        
        for posY in range(roadStartPositionY, roadEndPositionY + 1):
            for posX in range(roadStartPositionX, roadEndPositionX + 1):
                MAP[posY - 1][posX - 1] = 2 #1은 도로를 의
                    
    return np.array(MAP)

def show_plot(ax, Env, width, height):    
    position_BaseStation = Env.V2Ichannels.BS_position
    ax.set_ylim(-100, height +  100)
    ax.set_xlim(-100, width + 100)
    
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



Env = Environ(down_lanes,up_lanes,left_lanes,right_lanes, width, height, n_Veh) 
position_BaseStation = [width/2, height/2] 
Env.test_channel()

v2vlink_fastfading_normal = np.array(v2vlink_fastfading_normal)
v2ilink_fastfading_normal = np.array(v2ilink_fastfading_normal)
v2vlink_fastfading_rayleight = np.array(v2vlink_fastfading_rayleight)
v2ilink_fastfading_rayleight = np.array(v2ilink_fastfading_rayleight)

indexSize_normal = int(v2vlink_fastfading_normal.max() - v2vlink_fastfading_normal.min()) + 1
indexSize_reyleight = int(v2vlink_fastfading_rayleight.max() - v2vlink_fastfading_rayleight.min()) + 1

hist_normal = np.zeros(indexSize_normal)
hist_normalX = []

for normalIndex in range(len(v2vlink_fastfading_normal)):
    index = int(v2vlink_fastfading_normal[normalIndex] + abs(v2vlink_fastfading_normal.min()))
    hist_normal[index] = hist_normal[index] + 1

for normalIndex in range(indexSize_normal):
    hist_normalX.append(v2vlink_fastfading_normal.min() + normalIndex)


hist_reyleight = np.zeros(indexSize_reyleight)
hist_reyleightX = []

for reyleighlIndex in range(len(v2vlink_fastfading_rayleight)):
    index = int(v2vlink_fastfading_rayleight[reyleighlIndex] + abs(v2vlink_fastfading_rayleight.min()))
    hist_reyleight[index] = hist_reyleight[index] + 1
 
for reyleighlIndex in range(indexSize_reyleight):
    hist_reyleightX.append(v2vlink_fastfading_rayleight.min() + reyleighlIndex)
    
        
plt.plot(hist_reyleightX, hist_reyleight, label = 'Reyleigh')    
plt.plot(hist_normalX, hist_normal, label = 'Gaussian distribution')
plt.legend()
#plt.scatter(positionX_vehicle, positionY_vehicle, color = color ,label='vehicle', marker='x')
    
#Env.test_channel()    
