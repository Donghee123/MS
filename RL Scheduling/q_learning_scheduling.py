# -*- coding: utf-8 -*-
"""
Created on Thu May 20 16:58:50 2021

@author: CNL-B3
"""
import numpy as np
import random
import math 
import matplotlib.pyplot as plt

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
    
    def GetThroughput(self, inputSNR):
        return math.log2(1.0 + self.dB2real(inputSNR));
 
    
def CreateSNR(snrCreater,ueAVGSNRS):
    answerSNRs = []
    
    for AvgSNR in ueAVGSNRS:
        answerSNRs.append(snrCreater.GetReleighSNR(AvgSNR))
        
    return answerSNRs
    
"""
Q값을 저장 하고 정책에 따라 action을 수행
"""
class Q_LearningModel:
    
    def __init__(self, usercount, useEpsilon = True,  initepsilon = 0.5, alpha = 0.1, gamma = 0.1, updateEpsilon = 1.01):
        """        
        Parameters
        ----------
                     
            
        Returns
        -------
        none
        
        기능
        -------        
        모든 상태 변수 초기화
        """
        self.usercount = usercount
        self.QValueStore = {} 
        self.ThrouhputStore = {} 
        self.QSpaceSize = 0
        self.selectPair = []
        
        
        for i in range(usercount):
            self.QSpaceSize += usercount - i - 1
            for pair in range(i + 1, usercount):
                self.selectPair.append([i,pair])
                        
        
        self.initepsilon = initepsilon
        self.epsilon = initepsilon
        self.useEpsilon = useEpsilon
        self.alpha = alpha
        self.gamma = gamma
        self.updateEpsilon = updateEpsilon
        
        
        self.throughputs = 0
        
        
        self.fairness = 0
    
    #db값을 실수로
    def dB2real(self,fDBValue):
        """        
        Parameters
        ----------
        fDBValue : float                   
            
        Returns
        -------
        float  
        
        기능
        -------        
        db 값은 real 값으로 반환
        """
        return pow(10.0, fDBValue/10.0);

    def GetThroughput(self, inputSNR):
        """        
        Parameters
        ----------
        inputSNR : float                   
            
        Returns
        -------
        float  
        
        기능
        -------        
        SNR에 따른 처리율 반환
        """
        
        return math.log2(1.0 + self.dB2real(inputSNR));
    
    def GetFairness(self):
        """        
        Parameters
        ----------
        None                   
            
        Returns
        -------
        None  
        
        기능
        -------        
        현재 fairness 반환
        """
        return self.fairness
    
    def ResetEpsilon(self):
        """        
        Parameters
        ----------
        None                   
            
        Returns
        -------
        None  
        
        기능
        -------        
        입실론 값  초기화
        """
        self.epsilon = self.initepsilon
    
    def GetSelectUserPair(self,action):
        """        
        Parameters
        ----------
        action : int                       
            
        Returns
        -------
        pair
        
        기능
        -------        
        해당 action에서 선택하는 user 인덱스 반환
        """
        
        return self.selectPair[action][0], self.selectPair[action][1]
          
    def GetThrouputValue(self, state):
        """        
        Parameters
        ----------
        state : list                        
            
        Returns
        -------
        None
        
        기능
        -------        
        해당 state에서 throuhput값 불러오기
        """
        convertState = self.ConvertListToStringValue(state)
            
        if convertState in self.ThrouhputStore:
            return self.ThrouhputStore[convertState]
        else:
            return 0
        
    
    def SetThrouputValue(self, state, throuhput):
        """        
        Parameters
        ----------
        state : list            
        qValue : float
             
            
        Returns
        -------
        None
        
        기능
        -------        
        해당 state에서 throuhput값 저장
        """
        
        convertState = self.ConvertListToStringValue(state)
        self.ThrouhputStore[convertState] = throuhput
        
    def SetQValue(self,state, qValue):
        """        
        Parameters
        ----------
        state : list            
        qValue : List
                      
        Returns
        -------
        None
        
        기능
        -------        
        Q값 저장
        List 통째로 저장 하므로 비효율적임
        """
        
        convertState = self.ConvertListToStringValue(state)
        self.QValueStore[convertState] = qValue
        
    def GetQValue(self,state): 
        '''       
        Parameters
        ----------
        state : list
            

        Returns
        -------
        TYPE : np.array
            
        기능
        -------        
        Q값 불러오기
        List 통째로 반환 비효율적임 
        '''
        convertState = self.ConvertListToStringValue(state)
            
        if convertState in self.QValueStore:
            return self.QValueStore[convertState]
        else:
            return np.zeros(self.QSpaceSize)
        
    
    def ConvertListToStringValue(self,value):
        """
        Parameters
        ----------
        state : list
        
        Returns
        -------
        convertState : string   
        
        기능
        -------        
        List로 들어온 데이터 문자열로 반환 "값,값,값,값," 형식
        """
        convertState = ""
        
        for i in value:
            convertState += str(int(i)) + ','
            
        return convertState
    
    def ConvertStringToListValue(self,value):
        """
        Parameters
        ----------
        state : string
        
        Returns
        -------
        list  
        
        기능
        -------        
        들어온 문자열을 ',' 기준으로 구분 후 list 반환
        """
        return value.split[',']
        
    def GetReward(self, afterThrouhput, preThrouhput, afterFairness, preFairness):
        """
        Parameters
        ----------
        preThrouhput : float
        afterThrouhput : float
        preFairness : float
        afterFairness : float
        
        Returns
        -------
        float  
        
        기능
        -------        
        reward 계산
        """
        return (afterThrouhput - preThrouhput) + (afterFairness - preFairness)
    
    def CalThrouhput(self,SelSNR1, SelSNR2):
        """
        Parameters
        ----------
        SelSNR1 : float
        SelSNR2 : float
        
        Returns
        -------
        pair  
        
        기능
        -------        
        입력 SNR에 따른 Throughput 계산후 pair로 반환
        """
        return [self.GetThroughput(SelSNR1),self.GetThroughput(SelSNR2)]
        
    def SelectAction(self,state):
        """
        Parameters
        ----------
        state : list
        
        Returns
        -------
        choice : int  
        
        기능
        -------        
        입력 state를 보고 action을 결정
        epsilon 확률로 greedy
        1-epsilon 확률로 random
        """
        
        choice = 0
        
        if random.random() < self.epsilon:
            choice = np.argmax(self.GetQValue(state))
        else:
            choice = random.randrange(len(self.GetQValue(state)))
        
        self.epsilon *= self.updateEpsilon
        
        return choice
        
    def UpdateQValue(self, state, action, nextstate):
        """
        Parameters
        ----------
        state : string
        action : int
        nextstate : string
        
        Returns
        -------
        None
        
        기능
        -------        
        state, action, nextstate를 이용해 Reward 계산 후 
        q값 update
        """
        
        qValue = self.GetQValue(state)
        nextqValue = self.GetQValue(nextstate)
        
        selFirstSNR, selSecondSNR = self.GetSelectUserPair(action)
        
        #Reward 계산
        throughput1, throughput2 = self.CalThrouhput(state[selFirstSNR],state[selSecondSNR])    
        
        self.throughputs =  throughput1 + throughput2
        
        #현재 상태에서 저장하고 있는 최대 처리율 값 - 현재 상태에서 계산한 처리율값
        reward = self.GetReward(self.throughputs,self.GetThrouputValue(state),0,0)
        
       
        #새로 나온 throuput을값이 더 높다면 해당 상태의 throuput 저장
        if self.GetThrouputValue(state) < self.throughputs:
            self.SetThrouputValue(state, self.throughputs)   
            
        self.preFairness = self.fairness
        
        #Q-learning
        qValue[action] = qValue[action] + (self.alpha * (reward + self.gamma * nextqValue[np.argmax(nextqValue)] - qValue[action]))
        
        #update한 결과 저장
        self.SetQValue(state, qValue)
        
        
#simulation
ueCount = 4
updateEpsilon=1.00001
alpha = 0.4
gamma = 0.1
q_model = Q_LearningModel(usercount=ueCount, updateEpsilon=updateEpsilon, alpha=alpha, gamma=gamma)
snrCreater = SNRCreater()


ueAVGSNRS = []

#UE의 댓수 만큼 반복
for j in range(0,ueCount):
    #UE의 평균 snr 10~30 선정
    ueAVGSNRS.append(random.randrange(10,30))
        
q_model.ResetEpsilon()
throuput = []

episode = 1000
step = 100000

#episode
for i in range(episode):
    totalthroughputs = 0
    
    #step
    for j in range(step):
        
        ueOfSNRs = CreateSNR(snrCreater, ueAVGSNRS)
        action = q_model.SelectAction(ueOfSNRs)
        nextUEOfSNRs = CreateSNR(snrCreater, ueAVGSNRS)
        q_model.UpdateQValue(ueOfSNRs, action, nextUEOfSNRs)
        totalthroughputs += q_model.throughputs
    
    totalthroughputs /= step
    throuput.append(totalthroughputs)
        
plt.plot(throuput) 
plt.show()
    
        
        
        
        
        
        