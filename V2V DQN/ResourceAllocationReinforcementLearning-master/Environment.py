from __future__ import division
import numpy as np
import time
import random
import math
# This file is revised for more precise and concise expression.
"""
분석 순서
1. 차량의 Dropping model 찾기.
2. 5G Channel 모델 찾기
"""
class V2Vchannels:              
    # Simulator of the V2V Channels
    def __init__(self, n_Veh, n_RB):
        self.t = 0
        self.h_bs = 1.5
        self.h_ms = 1.5
        self.fc = 2
        self.decorrelation_distance = 10
        self.shadow_std = 3
        self.n_Veh = n_Veh
        self.n_RB = n_RB
        self.update_shadow([])
    def update_positions(self, positions):
        self.positions = positions
    def update_pathloss(self):
        self.PathLoss = np.zeros(shape=(len(self.positions),len(self.positions)))
        for i in range(len(self.positions)):
            for j in range(len(self.positions)):
                self.PathLoss[i][j] = self.get_path_loss(self.positions[i], self.positions[j])
    def update_shadow(self, delta_distance_list):
        delta_distance = np.zeros((len(delta_distance_list), len(delta_distance_list)))
        for i in range(len(delta_distance)):
            for j in range(len(delta_distance)):
                delta_distance[i][j] = delta_distance_list[i] + delta_distance_list[j]
        if len(delta_distance_list) == 0: 
            self.Shadow = np.random.normal(0,self.shadow_std, size=(self.n_Veh, self.n_Veh))
        else:
            self.Shadow = np.exp(-1*(delta_distance/self.decorrelation_distance)) * self.Shadow +\
                         np.sqrt(1 - np.exp(-2*(delta_distance/self.decorrelation_distance))) * np.random.normal(0, self.shadow_std, size = (self.n_Veh, self.n_Veh))
    def update_fast_fading(self):
        h = 1/np.sqrt(2) * (np.random.normal(size=(self.n_Veh, self.n_Veh, self.n_RB) ) + 1j * np.random.normal(size=(self.n_Veh, self.n_Veh, self.n_RB)))
        self.FastFading = 20 * np.log10(np.abs(h))
    def get_path_loss(self, position_A, position_B):
        d1 = abs(position_A[0] - position_B[0])
        d2 = abs(position_A[1] - position_B[1])
        d = math.hypot(d1,d2)+0.001
        d_bp = 4 * (self.h_bs - 1) * (self.h_ms - 1) * self.fc * (10**9)/(3*10**8)     
        def PL_Los(d):
            if d <= 3:
                return 22.7 * np.log10(3) + 41 + 20*np.log10(self.fc/5)
            else:
                if d < d_bp:
                    return 22.7 * np.log10(d) + 41 + 20 * np.log10(self.fc/5)
                else:
                    return 40.0 * np.log10(d) + 9.45 - 17.3 * np.log10(self.h_bs) - 17.3 * np.log10(self.h_ms) + 2.7 * np.log10(self.fc/5)
        def PL_NLos(d_a,d_b):
                n_j = max(2.8 - 0.0024*d_b, 1.84)
                return PL_Los(d_a) + 20 - 12.5*n_j + 10 * n_j * np.log10(d_b) + 3*np.log10(self.fc/5)
        if min(d1,d2) < 7: 
            PL = PL_Los(d)
            self.ifLOS = True
            self.shadow_std = 3
        else:
            PL = min(PL_NLos(d1,d2), PL_NLos(d2,d1))
            self.ifLOS = False
            self.shadow_std = 4                      # if Non line of sight, the std is 4
        return PL

class V2Ichannels: 
    # Simulator of the V2I channels
    def __init__(self, n_Veh, n_RB):
        self.h_bs = 25
        self.h_ms = 1.5        
        self.Decorrelation_distance = 50        
        self.BS_position = [750/2, 1299/2]    # Suppose the BS is in the center
        self.shadow_std = 8
        self.n_Veh = n_Veh
        self.n_RB = n_RB
        self.update_shadow([])
    def update_positions(self, positions):
        self.positions = positions
        
    def update_pathloss(self):
        self.PathLoss = np.zeros(len(self.positions))
        for i in range(len(self.positions)):
            d1 = abs(self.positions[i][0] - self.BS_position[0])
            d2 = abs(self.positions[i][1] - self.BS_position[1])
            distance = math.hypot(d1,d2) # change from meters to kilometers
            self.PathLoss[i] = 128.1 + 37.6*np.log10(math.sqrt(distance**2 + (self.h_bs-self.h_ms)**2)/1000)
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

"""
차량 
- 위치 선정 공식:  
- 방향 선정 공식:
- 속력 선정 공식:
"""
class Vehicle:
    # Vehicle simulator: include all the information for a vehicle
    def __init__(self, start_position, start_direction, velocity):
        self.position = start_position
        self.direction = start_direction
        self.velocity = velocity
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
        self.V2V_Shadowing = []     #v2v link의 Shadowing?
        self.V2I_Shadowing = []     #v2i link의 Shadowing?
        self.delta_distance = []    #?
        self.n_RB = 20              #resource block의 수 차량들이 주파수 점유 할 수 있는 수
        self.n_Veh = n_Veh             #최대 vehicle 수
        
        self.V2Vchannels = V2Vchannels(self.n_Veh, self.n_RB)  # V2V 채널 Class, 차량 수와 동일함.
        self.V2Ichannels = V2Ichannels(self.n_Veh, self.n_RB)

        self.V2V_Interference_all = np.zeros((self.n_Veh, 3, self.n_RB)) + self.sig2
        self.n_step = 0
        
    def add_new_vehicles(self, start_position, start_direction, start_velocity):    
        self.vehicles.append(Vehicle(start_position, start_direction, start_velocity))
        
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
            
            """
            
            ind = np.random.randint(0,len(self.down_lanes))
            start_position = [self.down_lanes[ind], random.randint(0,self.height)]
            start_direction = 'd'
            #self.add_new_vehicles(start_position,start_direction,random.randint(10,15))
            self.add_new_vehicles(start_position,start_direction,36)
            
            start_position = [self.up_lanes[ind], random.randint(0,self.height)]
            start_direction = 'u'
            #self.add_new_vehicles(start_position,start_direction,random.randint(10,15))
            self.add_new_vehicles(start_position,start_direction,36)
            
            start_position = [random.randint(0,self.width), self.left_lanes[ind]]
            start_direction = 'l'
            #self.add_new_vehicles(start_position,start_direction,random.randint(10,15))
            self.add_new_vehicles(start_position,start_direction,36)
            
            start_position = [random.randint(0,self.width), self.right_lanes[ind]]
            start_direction = 'r'
            #self.add_new_vehicles(start_position,start_direction,random.randint(10,15))
            self.add_new_vehicles(start_position,start_direction,36)
            
        self.V2V_Shadowing = np.random.normal(0, 3, [len(self.vehicles), len(self.vehicles)])
        self.V2I_Shadowing = np.random.normal(0, 8, len(self.vehicles))
        self.delta_distance = np.asarray([c.velocity for c in self.vehicles])
        #self.renew_channel()
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
            if self.vehicles[i].direction == 'u':
                #print ('len of position', len(self.position), i)
                for j in range(len(self.left_lanes)):
                    
                    if (self.vehicles[i].position[1] <=self.left_lanes[j]) and ((self.vehicles[i].position[1] + delta_distance) >= self.left_lanes[j]):   # came to an cross
                        if (random.uniform(0,1) < 0.4):
                            self.vehicles[i].position = [self.vehicles[i].position[0] - (delta_distance - (self.left_lanes[j] - self.vehicles[i].position[1])),self.left_lanes[j] ] 
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
        self.n_step = 0
        self.vehicles = []
        n_Veh = 20
        self.n_Veh = n_Veh
        self.add_new_vehicles_by_number(int(self.n_Veh/4))
        step = 1000
        time_step = 0.1  # every 0.1s update
        for i in range(step):
            self.renew_positions() 
            positions = [c.position for c in self.vehicles]
            self.update_large_fading(positions, time_step)
            self.update_small_fading()
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
        self.V2Ichannels.update_positions(positions)
        self.V2Vchannels.update_positions(positions)
        self.V2Ichannels.update_pathloss()
        self.V2Vchannels.update_pathloss()
        delta_distance = time_step * np.asarray([c.velocity for c in self.vehicles])
        self.V2Ichannels.update_shadow(delta_distance)
        self.V2Vchannels.update_shadow(delta_distance)
    def update_small_fading(self):
        self.V2Ichannels.update_fast_fading()
        self.V2Vchannels.update_fast_fading()
        
    def renew_neighbor(self):   
        # ==========================================
        # update the neighbors of each vehicle.
        # 각 차량들의 거리를 보고 현재 차량의 vehicle class의 neighbor를 재갱신한다.
        # ===========================================
        for i in range(len(self.vehicles)):
            self.vehicles[i].neighbors = []
            self.vehicles[i].actions = []
            #print('action and neighbors delete', self.vehicles[i].actions, self.vehicles[i].neighbors)
        Distance = np.zeros((len(self.vehicles),len(self.vehicles)))
        z = np.array([[complex(c.position[0],c.position[1]) for c in self.vehicles]])
        Distance = abs(z.T-z)
        """
        Distance = [차량수 x 차량수]
        Distance i : 송신차량
        Distance j : 수신 차량
        Distance[i][j] = 두차량의 거리, Distance[j][i]와 같음
        Distance[i][i] = 자기자신 -> 0        
        """
        for i in range(len(self.vehicles)):       
            sort_idx = np.argsort(Distance[:,i])
            for j in range(3):
                """
                현재 차량 vehicles[i]에서 가장 가까운 차량 3개를 선택함.
                """
                self.vehicles[i].neighbors.append(sort_idx[j+1])
                
            """
            차량의수 / 5 개 만큼 송신하고자 하는 차량을 3대 선택함.
            40대 차량 기준 -> 40/5 -> 8대 차량중 랜덤 선택
            8대 차량 구성 : 인접한 차량 3대 + 덜 인접한 차량 5대
            8대 중 3대 랜덤 선택            
            """
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
        #
        # self.V2V_channels_abs : ( Pathloss + Shadow ) vehicles x vehicles x resource block
        # self.V2I_channels_abs : ) Pathloss + Shadow ) vehicles x resource block
        #
        # self.V2V_channels_with_fastfading : self.V2V_channels_abs + fast fading
        # self.V2I_channels_with_fastfading : self.V2I_channels_abs + fast fading
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
        self.demand -= V2V_Rate * self.update_time_test * 1500    # decrease the demand, V2V Link에서 요구하는 데이터 량
        self.test_time_count -= self.update_time_test               # compute the time left for estimation
        self.individual_time_limit -= self.update_time_test         # compute the time left for individual V2V transmission
        self.individual_time_interval -= self.update_time_test      # compute the time interval left for next transmission

        # --- update the demand ---
        
        new_active = self.individual_time_interval <= 0
        self.activate_links[new_active] = True
        self.individual_time_interval[new_active] = np.random.exponential(0.02, self.individual_time_interval[new_active].shape ) + self.V2V_limit
        self.individual_time_limit[new_active] = self.V2V_limit
        self.demand[new_active] = self.demand_amount #
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
        Interference = np.zeros(self.n_RB)   # Calculate the interference from V2V to V2I     V2I의 간섭신호를 계산함. V2I Interference = V2V 간섭 신호 + V2I 간섭 신호
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
        actions[(np.logical_not(self.activate_links))] = -1   #들어온 action에서 동일한 리소스블럭을 사용하는 V2V의 간섭 신호들을 더함.
        for i in range(self.n_RB):
            indexes = np.argwhere(actions == i) #indexes [17, 1] -> 17번번째 차량이 1번번째 차량에게 데이터를 전송하는 것임. -> 현재 action과 동일한 리소스 블럭을 사용하는 V2V Link
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
                    Interfence_times[indexes[j,0],indexes[j,1]] += 1 # 필요 없는듯?
                    Interfence_times[indexes[k,0],indexes[k,1]] += 1 # 필요 없는듯?            

        self.V2V_Interference = V2V_Interference + self.sig2 #위의 반복문에서 계산한 V2V 간섭신호들을 정함.
        V2V_Rate = np.log2(1 + np.divide(V2V_Signal, self.V2V_Interference)) # V2V Signal / V2V Interference -> 현재 차량에서한 action의 V2V Link의 SINR을 계산함 -> 3개의 V2V Rate가 나옴 -> 이웃차량이 3대이기 때문.
        V2I_Signals = self.V2I_power_dB-self.V2I_channels_abs[0:min(self.n_RB,self.n_Veh)] + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure # V2I power는 23 dB 고정, 모든 차량에 대한 Signal이 나옴
        V2I_Rate = np.log2(1 + np.divide(10**(V2I_Signals/10), self.V2I_Interference[0:min(self.n_RB,self.n_Veh)])) #V2I Signal / V2V Interference -> V2I Link의 SINR을 계산함
        #print("V2I information", V2I_Signals, self.V2I_Interference, V2I_Rate)
        
        # -- compute the latency constraits --
        self.demand -= V2V_Rate * self.update_time_asyn * 1500    # decrease the demand, 계산된 V2V_Rate를 보고 차량들의 요구하는 비트수를 감소 시킴. 즉 일정 SINR을 가지고 데이터 전송을 의미함.
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
        early_finish = np.multiply(self.demand <= 0, self.activate_links) #데이터 전송을 제한시간안에  모두 마친 링크
        unqulified = np.multiply(self.individual_time_limit <=0, self.activate_links)#데이터 전송을 제한시간 안에  마친 링크
        self.activate_links[np.add(early_finish, unqulified)] = False #데이터 전송을 제한 시간안에 모두 마치거나, 제한 시간안에 못보낸 링크들을 activate link에서 False로 link를 재활성화 시킴.
        self.success_transmission += np.sum(early_finish) #데이터 전송을 마친 경우에 한해 success_transmission에 더함
        self.failed_transmission += np.sum(unqulified)  #데이터 전송을 못 마친 경우에 한해 failed_transmission에 더함
        fail_percent = self.failed_transmission/(self.failed_transmission + self.success_transmission + 0.0001) #두 확률을 계산하여 실패확률을 계산함. fail_percent는 제한시간안에 요구한 데이터 량만큼 처리 했는지를 의미함.    
        return V2I_Rate, fail_percent

    def Compute_Performance_Reward_Batch(self, actions_power, idx):    # add the power dimension to the action selection
        # ==================================================
        # ------------- Used for Training ----------------
        # ==================================================
        # 선택한 resource block
        actions = actions_power.copy()[:,:,0] 
        # 선택한 power level          
        power_selection = actions_power.copy()[:,:,1]   
        
        print('select resource block')
        print(actions)
        print('select power level')
        print(power_selection)
        #모든 차량에 대해서 연결된 3개의 차량의 Interference
        V2V_Interference = np.zeros((len(self.vehicles), 3))
        
        #모든 차량에 대해서 연결된 3개의 차량의 신호
        V2V_Signal = np.zeros((len(self.vehicles), 3))
        
        
        Interfence_times = np.zeros((len(self.vehicles), 3))    #  3 neighbors
        
        #print(actions)
        
        #idx[0] 전송차량, idx[1] 수신차량
        origin_channel_selection = actions[idx[0], idx[1]]
        
        actions[idx[0], idx[1]] = 100  # something not relavant
        
        for i in range(self.n_RB):
            
            #indexes : i번째 resource block을 사용하는 송신 차량, 수신 차량 pair을 찾음
            indexes = np.argwhere(actions == i)
            #print('index',indexes)
            # i번째 리소스 블록을 사용하는 송신 차량, 수신 차량 pair에 대해서 계산함.
            # V2V_Signal은 송신 차량의 전송 신호 데이터
            # V2V_Interference는 같은 리소스 블록을 사용하기때문에 상호간섭을 누적시킴. 
            # 즉 같은 리소스 블록을 많이 쓰면 interference가 증가됨
            for j in range(len(indexes)):
                
                #receiver_j = self.vehicles[indexes[j,0]].neighbors[indexes[j,1]]
                
                #i번째 resource block을 사용하는 신호의 수신 차량
                receiver_j = self.vehicles[indexes[j,0]].destinations[indexes[j,1]] 
                
                # indexes[j, 0] : 현재 차량 인덱스
                # indexes[j, 1] : 현재 차량 기준으로 선택한 차량 인덱스
                
                
                # V2V_Signal = 10 ^ ( (선택한 파워 dbm - 선택한 리소스 블록의 fading값 + 2 * 차량의 안테나 게인 - 9(차량 노이즈)) / 10)
                
                # V2V Signal은 송신 차량, 수신 차량으로 구분지어서 저장됨. 
                V2V_Signal[indexes[j, 0],indexes[j, 1]] = 10**((self.V2V_power_dB_List[power_selection[indexes[j, 0],indexes[j, 1]]] - self.V2V_channels_with_fastfading[indexes[j,0], receiver_j, i]+ 2 * self.vehAntGain - self.vehNoiseFigure) / 10) 
                
                # 같은 리소스 블록을 사용하는 신호들의 V2V_Interference를 누적 시킴.
                V2V_Interference[indexes[j,0],indexes[j,1]] += 10**((self.V2I_power_dB - self.V2V_channels_with_fastfading[i,receiver_j,i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)  # interference from the V2I links
                
                # V2V_Interference 자세한 설명 
                # 1. i번째 Resource block에서 V2I link의 power_db와 송신 차량(j,0) -> 수신 차량 (j,1)에 대한 Fast fading에 영향을 미침 Interference 증가
                # 2. i번째 Resource block에서 다른 송신 차량(k,0)들이 보내는 신호와 수신 차량(j,1)이 받는 신호의 power_db((k,0) -> (k,1)) - Fast fading((k,0) -> (j,1))에 영향을 미침 Interference 증가
                # 3. i번째 Resource block에서 다른 수신 차량(k,1)들에 대해서도 현재 송신 차량(j,0)의 신호에 의한 power_db((j,0) -> (j,1)) - Fast fading((j,0) -> (k,1))에 영향을 미침 Interference 증가
                for k in range(j+1, len(indexes)):
                    receiver_k = self.vehicles[indexes[k,0]].destinations[indexes[k,1]]
                    V2V_Interference[indexes[j,0],indexes[j,1]] += 10**((self.V2V_power_dB_List[power_selection[indexes[k,0],indexes[k,1]]] - self.V2V_channels_with_fastfading[indexes[k,0],receiver_j, i] + 2 * self.vehAntGain - self.vehNoiseFigure)/10)
                    V2V_Interference[indexes[k,0],indexes[k,1]] += 10**((self.V2V_power_dB_List[power_selection[indexes[j,0],indexes[j,1]]] - self.V2V_channels_with_fastfading[indexes[j,0],receiver_k, i] + 2 * self.vehAntGain - self.vehNoiseFigure)/10)
                    Interfence_times[indexes[j,0],indexes[j,1]] += 1
                    Interfence_times[indexes[k,0],indexes[k,1]] += 1
         
        #계산한 V2V_Interference에서 sig2 노이즈를 더함.
        self.V2V_Interference = V2V_Interference + self.sig2
        
        """
        여기까지 정리
        2021/09/08
        1. V2V Signal 계산 (power, fast fading, 차량 antena gain, Noise)
        2. V2V Interference / time 계산  (power, fast fading, 차량 antena gain, Noise)
           - 동일한 resource block에서 V2I link, 다른 차량들의 V2V link를 고려한 interference 계산
           - 한번 계산 할때마다 1씩 시간 증가
        """
        
        V2V_Rate_list = np.zeros((self.n_RB, len(self.V2V_power_dB_List)))   # 3 of power level, V2V link Channel capacity를 저장함.
        Deficit_list = np.zeros((self.n_RB, len(self.V2V_power_dB_List)))
        
        for i in range(self.n_RB):
            indexes = np.argwhere(actions == i)
            V2V_Signal_temp = V2V_Signal.copy()            
            #receiver_k = self.vehicles[idx[0]].neighbors[idx[1]]
            receiver_k = self.vehicles[idx[0]].destinations[idx[1]]
            for power_idx in range(len(self.V2V_power_dB_List)):
                V2V_Interference_temp = V2V_Interference.copy()
                V2V_Signal_temp[idx[0],idx[1]] = 10**((self.V2V_power_dB_List[power_idx] - self.V2V_channels_with_fastfading[idx[0], self.vehicles[idx[0]].destinations[idx[1]],i] + 2*self.vehAntGain - self.vehNoiseFigure )/10)
                V2V_Interference_temp[idx[0],idx[1]] +=  10**((self.V2I_power_dB - self.V2V_channels_with_fastfading[i,self.vehicles[idx[0]].destinations[idx[1]],i] + 2*self.vehAntGain - self.vehNoiseFigure)/10)
                for j in range(len(indexes)):
                    receiver_j = self.vehicles[indexes[j,0]].destinations[indexes[j,1]]
                    V2V_Interference_temp[idx[0],idx[1]] += 10**((self.V2V_power_dB_List[power_selection[indexes[j,0], indexes[j,1]]] - self.V2V_channels_with_fastfading[indexes[j,0],receiver_k, i] + 2*self.vehAntGain - self.vehNoiseFigure)/10)
                    V2V_Interference_temp[indexes[j,0],indexes[j,1]] += 10**((self.V2V_power_dB_List[power_idx] - self.V2V_channels_with_fastfading[idx[0],receiver_j, i] + 2*self.vehAntGain - self.vehNoiseFigure)/10)
                V2V_Rate_cur = np.log2(1 + np.divide(V2V_Signal_temp, V2V_Interference_temp))#V2V link의 SINR 계산.
                if (origin_channel_selection == i) and (power_selection[idx[0], idx[1]] == power_idx):
                    V2V_Rate = V2V_Rate_cur.copy()
                V2V_Rate_list[i, power_idx] = np.sum(V2V_Rate_cur)
                Deficit_list[i,power_idx] = 0 - 1 * np.sum(np.maximum(np.zeros(V2V_Signal_temp.shape), (self.demand - self.individual_time_limit * V2V_Rate_cur * 1500)))
        Interference = np.zeros(self.n_RB)  
        V2I_Rate_list = np.zeros((self.n_RB,len(self.V2V_power_dB_List)))    # 3 of power level, V2I link Channel capacity를 저장함.
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
        
        #각기 다른 차량들의 time_limit을 1사이클만큼 시간이 지남을 의미       
        self.individual_time_limit -= self.update_time_train
        
        #time_limit이 0보다 같거나 작은 것이 있는 경우...??
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
                        V2V_Interference[k, m, i] += 10 ** ((self.V2I_power_dB - self.V2V_channels_with_fastfading[i][self.vehicles[k].destinations[m]][i] + 2 * self.vehAntGain - self.vehNoiseFigure)/10)
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
        # idx : 송신 차량, 수신 차량 정보를 가짐.
        # action : 선택한 power level, 선택한 resource block을 가짐.
        # ===========================================
        
        #reward list를 정의함 -> resource block 사이즈를 가짐
        rewards_list = np.zeros(self.n_RB)
        
        #action을 복사함.
        action_temp = actions.copy()
        
        #활성화된 link를 의미함
        #모든 차량기준으로 3개의 v2v link를 의미
        self.activate_links = np.ones((self.n_Veh,3), dtype = 'bool')
        
        #V2I reward(capacity), V2V reward(capacity), Time reward를 의미, 논문에서는 이 3개의 list를 각각 다른 weight로 선정하여 학습 시킴.
        V2I_rewardlist, V2V_rewardlist, time_left = self.Compute_Performance_Reward_Batch(action_temp,idx)
        
        
        self.renew_positions()
        self.renew_channels_fastfading()
        self.Compute_Interference(actions) 
        rewards_list = rewards_list.T.reshape([-1])
        V2I_rewardlist = V2I_rewardlist.T.reshape([-1])
        V2V_rewardlist = V2V_rewardlist.T.reshape([-1])
        V2I_reward = (V2I_rewardlist[actions[idx[0],idx[1], 0]+ 20*actions[idx[0],idx[1], 1]] - np.min(V2I_rewardlist))/(np.max(V2I_rewardlist) -np.min(V2I_rewardlist) + 0.000001)
        V2V_reward = (V2V_rewardlist[actions[idx[0],idx[1], 0]+ 20*actions[idx[0],idx[1], 1]] - np.min(V2V_rewardlist))/(np.max(V2V_rewardlist) -np.min(V2V_rewardlist) + 0.000001)
        lambdda = 0.1
        #print ("Reward", V2I_reward, V2V_reward, time_left)
        t = lambdda * V2I_reward + (1-lambdda) * V2V_reward
        #print("time left", time_left)
        #return t
        return t - (self.V2V_limit - time_left)/self.V2V_limit
    #모든 차량이 선택을 하면 renew_position, renew_channels_fastfading()를 함 -> 채널 재갱신
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

if __name__ == "__main__":
    
    up_lanes = [3.5/2,3.5/2 + 3.5,250+3.5/2, 250+3.5+3.5/2, 500+3.5/2, 500+3.5+3.5/2]
    down_lanes = [250-3.5-3.5/2,250-3.5/2,500-3.5-3.5/2,500-3.5/2,750-3.5-3.5/2,750-3.5/2]
    left_lanes = [3.5/2,3.5/2 + 3.5,433+3.5/2, 433+3.5+3.5/2, 866+3.5/2, 866+3.5+3.5/2]
    right_lanes = [433-3.5-3.5/2,433-3.5/2,866-3.5-3.5/2,866-3.5/2,1299-3.5-3.5/2,1299-3.5/2]
    width = 750
    height = 1299
    Env = Environ(down_lanes,up_lanes,left_lanes,right_lanes, width, height) 
    Env.test_channel()    
