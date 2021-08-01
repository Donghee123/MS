# -*- coding: utf-8 -*-
"""
Created on Fri May 14 00:39:15 2021

@author: Handonghee
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 10 22:06:09 2021

@author: Handonghee
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from math import floor
import math

#collisions 충돌 핸들링용 클래스
class IHT:
    "Structure to handle collisions"
    def __init__(self, size_val):
        self.size = size_val
        self.overfull_count = 0
        self.dictionary = {}

    def count(self):
        return len(self.dictionary)

    def full(self):
        return len(self.dictionary) >= self.size

    def get_index(self, obj, read_only=False): 
        #해쉬테이블 안에 값이 있으면 값 출력 -> 매우 많은 value function 값 리턴
        if obj in self.dictionary:
            return self.dictionary[obj]
        #ready_only이고 값이 없다면 None 리턴
        elif read_only:
            return None
        
        #read_only가 아니라면
        
        size = self.size
        count = self.count()
        
        #현재 count가 size보다 많으면
        if count >= size:
            if self.overfull_count == 0: print('IHT full, starting to allow collisions')
            self.overfull_count += 1
            return hash(obj) % self.size
        else:
            self.dictionary[obj] = count
            return count


def hash_coords(coordinates, m, read_only=False):
    if isinstance(m, IHT):
        return m.get_index(tuple(coordinates), read_only)
    if isinstance(m, int): 
        return hash(tuple(coordinates)) % m
    if m is None: 
        return coordinates

#floats[0]는 0 ~ 7의 사이의값 실제 position을 tile의 세계로 scaling으로 보냄
#floats[1]                   실제 velocity를 tile의 세계로 scaling으로 보냄
def tiles(iht_or_size, num_tilings, floats, ints=None, read_only=False):
    """returns num-tilings tile indices corresponding to the floats and ints"""
    if ints is None:
        ints = []
        
    qfloats = [floor(f * num_tilings) for f in floats]
    tiles = []
    
    # 0~7까지 반복
    for tiling in range(num_tilings):
        tilingX2 = tiling * 2
        coords = [tiling]
        b = tiling
        for q in qfloats:
            coords.append((q + b) // num_tilings)
            b += tilingX2
        coords.extend(ints)
        tiles.append(hash_coords(coords, iht_or_size, read_only))
    return tiles

# Tile coding ends
#######################################################################


EPSILON = 0.2
REWARD = -1

#선택한 유저
Sel_UE1 = 0
Sel_UE2 = 1
Sel_UE3 = 2
Sel_UE4 = 3

#액션 리스트
ACTIONS = [Sel_UE1, Sel_UE2, Sel_UE3,Sel_UE4]


#모든 SNR의 MIN, MAX값
SNR_MIN = -50
#맵에서 오른쪽 끝 
SNR_MAX = 50


class ValueFunction:
    
    def __init__(self, step_size, num_of_tilings=100, max_size=4048):
        
        #state을 구분하는 최대 갯수, 최대 4048 갯수
        self.max_size = max_size
        
        #실제로 state를 구분한 갯수, 100개 -> 100개를 function appoximation을 통해 continouse 한 것과 비슷하게 만들 것
        self.num_of_tilings = num_of_tilings

        # divide step size equally to each tiling ? 
        self.step_size = step_size / num_of_tilings

        # 해쉬 테이블을 이용해 Value function값을 저장할 것. key : 현재 SNR값, value : value function값
        self.hash_table = IHT(max_size)

        # 가중치 변수 w를 4048 0으로 생성
        self.weights = np.zeros(max_size)

        # 1step 포지션의 scale = 크게 구분한 갯수 / 맵의 실 거리 범위
        self.SNR_scale = self.num_of_tilings / (SNR_MAX - SNR_MIN)



    def get_active_tiles(self, inputSNRs, action):
        
        scalinginputSNRs =  [SNR * self.SNR_scale  for SNR in inputSNRs]
                
        active_tiles = tiles(self.hash_table, self.num_of_tilings, scalinginputSNRs , [action])
        
        return active_tiles

    # 현재 4개의 SNR을 받아 
    # value function 획득
    def value(self, inputSNRs, action):
        
        active_tiles = self.get_active_tiles(inputSNRs, action)
        
        #active_tiles에 해당하는 인덱스의 값만 summation         
        outputNode = np.sum(self.weights[active_tiles]) 
        
        return outputNode 

    # 가중치 업데이트 
    def learn(self, inputSNRs, action, target):
        #위치, 속도, 해당 위치와 속도에서 수행한 action, 그에 따른 보상
        
        #사용했던 가중치 리스트 획득
        active_tiles = self.get_active_tiles(inputSNRs, action)
        
        #사용했던 가중치들의 summation 
        estimation = np.sum(self.weights[active_tiles])
        
        #td target - 사용했던 가중치들의 summation 
        # w의 gradient
        delta = self.step_size * (target - estimation)
        
        for active_tile in active_tiles:
            self.weights[active_tile] += delta # gradient descent

    """ 임시로 안씀
    # get # of steps to reach the goal under current state value function
    def cost_to_go(self, position, velocity):
        costs = []
        for action in ACTIONS:
            costs.append(self.value(position, velocity, action))
        return -np.max(costs)
    """
    
# get action at @position and @velocity based on epsilon greedy policy and @valueFunction
# 0~3 중에 1개 선택
def get_action(inputSNRs, value_function):
    
    #액션은 Epsilon greedy 확률로 Random or greedy
    if np.random.binomial(1, EPSILON) == 1:
        return np.random.choice(ACTIONS) #ACTIONS = [-1,0,1]
    
    values = []
    
    #action은 -0.7 ~ 0.7 사이의 속력
    for action in ACTIONS:
        values.append(value_function.value(inputSNRs, action))
        
    return np.random.choice([action_ for action_, value_ in enumerate(values) if value_ == np.max(values)]) - 1

def getSNRFourUser():
    """
    채널별 랜덤 SNR 4개 취득
    """
    return [1,1,1,1]

def getReward(inputSNRs, selnumber):
    """
    선택한 SNR에 대해서 througput 연산후 return reward
    """
    reward = 0 #Reward 계산 할 것
    return reward


# semi-gradient n-step Sarsa
# n - step sarsa
def semi_gradient_n_step_sarsa(value_function, n=1):
    
    """
    랜덤 SNR 4개로 데이터 시작
    """
    current_SNRs = getSNRFourUser()
         
    # 현재 상태의 위치, 속력을 value_function과 비교하여 다음 action 획득
    current_action = get_action(current_SNRs, value_function)

    # track previous position, velocity, action and reward
    historyOfSNRS = [current_SNRs]
    historyOfAction = [current_action]   
    historyOfRewards = [getReward(current_SNRs, current_action)]

    # track the time
    time = 0

    # the length of this step
    T = 5000000
    while True:
        # go to next time step
        time += 1

        #T는 내가 정하는 반복 횟수
        if time < T:
            
            #새로운 SNR 취득
            current_SNRs = getSNRFourUser() 
            #SNR값 선택
            current_action = get_action(current_SNRs, value_function)
            #선택한 SNR값 기반 Reward 계산
            reward = getReward(current_SNRs, current_action)
            
            # 모든 히스토리 저장
            historyOfSNRS.append(current_SNRs)
            historyOfAction.append(current_action)   
            historyOfRewards.append(reward)
                           
            
            # get the time of the state to update
            update_time = time - n
            
        if update_time >= 0:
            returns = 0.0
                        
            # culmulative reward 계산
            for t in range(update_time + 1, min(T, update_time + n) + 1):
                returns += historyOfRewards[t]
                
            # culmulative reward에 state value function을 더함
            # culmulative reward + gamma * qˆ(S0, A0, w) = td target
            if update_time + n <= T:
                #culmulative reward += 업데이트 전 state value function 값 = td target
                returns += value_function.value(historyOfSNRS[update_time + n],
                                                historyOfAction[update_time + n])
                
            value_function.learn(historyOfSNRS[update_time], historyOfAction[update_time], returns)
                
        if update_time == T - 1:
            break
        

    return time


"""
# print learned cost to go
def print_cost(value_function, episode, ax):
    grid_size = 40
    positions = np.linspace(MAP_LEFT, MAP_RIGHT, grid_size)
    # positionStep = (MAP_RIGHT - MAP_LEFT) / grid_size
    # positions = np.arange(MAP_LEFT, MAP_RIGHT + positionStep, positionStep)
    # velocityStep = (VELOCITY_RIGHT - VELOCITY_LEFT) / grid_size
    # velocities = np.arange(VELOCITY_LEFT, VELOCITY_RIGHT + velocityStep, velocityStep)
    velocities = np.linspace(VELOCITY_LEFT, VELOCITY_RIGHT, grid_size)
    
    axis_x = []
    axis_y = []
    axis_z = []
    
    for position in positions:
        for velocity in velocities:
            axis_x.append(position)
            axis_y.append(velocity)
            axis_z.append(value_function.cost_to_go(position, velocity))

    ax.scatter(axis_x, axis_y, axis_z)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Cost to go')
    ax.set_title('Episode %d' % (episode + 1))
"""
# simulation1. episode 1, 12, 104, 1000, 9000번의 3d 그래프 저장

def start_simulation1():
    
    #수행할 에피소드 설정 : 9000번
   
    #에피소드 별 Valuefunction 그래프 1번, 12번 104번, 1000번, 9000번
    plot_episodes = [0, 11, 103, 999, 8999]
        
    #plot 설정
    fig = plt.figure(figsize=(40, 10))
    axes = [fig.add_subplot(1, len(plot_episodes), i+1, projection='3d') for i in range(len(plot_episodes))]
    
    #value function 초기화
    num_of_tilings = 8
    alpha = 0.3
    value_function = ValueFunction(alpha, num_of_tilings)
    
    
    for ep in tqdm(range(plot_episodes[len(plot_episodes) - 1] + 1)):
        semi_gradient_n_step_sarsa(value_function)
        #if ep in plot_episodes:
            #print_cost(value_function, ep, axes[plot_episodes.index(ep)])

    #print_cost(value_function, ep, axes[plot_episodes.index(ep)])

    plt.savefig('../images/figure_10_1.png')
    plt.close()
    
    
# simulation2. alpha 값 0.1, 0.2, 0.5 변경하면서 테스트 후 2d 그래프 저장
def start_simulation2():
      
    #value function 초기화
    num_of_tilings = 8
    alphas = [0.1,0.2,0.5]   
    alphasstepPerEpisods = [] 
    x = []
    
    for i in range(1000):
        x.append(i + 1)
            
    #에피소드 1번당 수행하는 timestep 저장
    for alpha in alphas:
        value_function = ValueFunction(alpha, num_of_tilings)
        stepPerEpisodes = []
        for i in range(1000):
            stepPerEpisodes.append(semi_gradient_n_step_sarsa(value_function))
            
       
        alphasstepPerEpisods.append(stepPerEpisodes)
    
    plt.xlabel('Episode')
    plt.ylabel('StepPerEpisode')
    
    plt.plot(x, alphasstepPerEpisods[0], label='alpha : 0.1')
    plt.plot(x, alphasstepPerEpisods[1], label='alpha : 0.2')
    plt.plot(x, alphasstepPerEpisods[2], label='alpha : 0.5')
    
    plt.ylim(100,1000)
    plt.xlim(0,500)
    
    plt.legend(loc='best', ncol=2) 
    plt.show()
    
# simulation3. Bootstrap 1,8 변경하면서 테스트 후 2d 그래프 저장
def start_simulation3():
      
    #value function 초기화
    num_of_tilings = 100
    alpha = 0.3
    bootstraps = [1,8]
    bootstrapsstepPerEpisods = [] 
    x = []
    
    for i in range(1000):
        x.append(i + 1)
            
    #에피소드 1번당 수행하는 timestep 저장
    for bootstrap in bootstraps:
        value_function = ValueFunction(alpha, num_of_tilings)
        stepPerEpisodes = []
        for i in range(1000):
            stepPerEpisodes.append(semi_gradient_n_step_sarsa(value_function, bootstrap))
                  
        bootstrapsstepPerEpisods.append(stepPerEpisodes)
    
    plt.xlabel('Episode')
    plt.ylabel('StepPerEpisode')
    
    plt.plot(x, bootstrapsstepPerEpisods[0], label='n : 1')
    plt.plot(x, bootstrapsstepPerEpisods[1], label='n : 8')

    
    plt.ylim(100,1000)
    plt.xlim(0,500)
    
    plt.legend(loc='best', ncol=2) 
    plt.show()
    
#start_simulation1()    
#start_simulation2()
start_simulation3()





