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
    
    def __init__(self, usercount, useEpsilon = True,  initepsilon = 0.5, alpha = 0.1, gamma = 0.1):
        """
        Init
        QValueStore
        epsilon
        useEpsilon
        alpha
        gamma
        """
        self.usercount = usercount
        self.QValueStore = {} 
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
        
        self.preThroughput = 0
        self.throughputs = 0
        
        self.preFairness = 0
        self.fairness = 0
    
    #db값을 실수로
    def dB2real(self,fDBValue):
        return pow(10.0, fDBValue/10.0);

    def GetThroughput(self, inputSNR):
        return math.log2(1.0 + self.dB2real(inputSNR));
    
    def GetFairness(self):
        return self.fairness
    
    def ResetEpsilon(self):
        self.epsilon = self.initepsilon
    
    def GetSelectUserPair(self,action):
        return self.selectPair[action][0], self.selectPair[action][1]
    
    def SetQValue(self,state, qValue):
        """
        

        Parameters
        ----------
        state : list
            DESCRIPTION.
        qValue : List
            DESCRIPTION.
        Returns
        -------
        None.

        """
        
        convertState = self.ConvertListToStringValue(state)
        self.QValueStore[convertState] = qValue
        
    def GetQValue(self,state): 
        '''
        

        Parameters
        ----------
        state : list
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

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

        """
        return value.split[',']
        
    def GetReward(self,preThrouhput, afterThrouhput, preFairness, afterFairness):
        return (afterThrouhput - preThrouhput) + (afterFairness - preFairness)
    
    def CalThrouhput(self,SelSNR1, SelSNR2):
        """
       
        Parameters
        ----------
        SelSNR1 : float
            선택한 단말1의 SNR
        SelSNR2 : float
            선택한 단말2의 SNR

        Returns
        -------
        list
            DESCRIPTION.

        """
        
        return [self.GetThroughput(SelSNR1),self.GetThroughput(SelSNR2)]
        
    def SelectAction(self,state):
        """
        epsilon-greedy 확률로 greedy or random

        Parameters
        ----------
        state : list
            
        Returns
        -------
        selectiveIndex : int
            
        """
        choice = 0
        
        if random.random() < self.epsilon:
            choice = np.argmax(self.GetQValue(state))
        else:
            choice = random.randrange(len(self.GetQValue(state)))
        
        self.epsilon *= 1.00001
        
        return choice
        
    def UpdateQValue(self, state, action, nextstate):
        
        qValue = self.GetQValue(state)
        nextqValue = self.GetQValue(nextstate)
        
        selFirstSNR, selSecondSNR = self.GetSelectUserPair(action)
        
        #Reward 계산
        throughput1, throughput2 = self.CalThrouhput(state[selFirstSNR],state[selSecondSNR])    
        
        self.throughputs =  throughput1 + throughput2
        reward = self.GetReward(self.preThroughput,self.throughputs,0,0)
        
        #이전 bps, fairness 저장
        self.preThroughput = self.throughputs
        self.preFairness = self.fairness
        
        #Q-learning
        qValue[action] = qValue[action] + (self.alpha * (reward + self.gamma * nextqValue[np.argmax(nextqValue)] - qValue[action]))
        
        #update한 결과 저장
        self.SetQValue(state, qValue)
        
        
#simulation
ueCount = 4

q_model = Q_LearningModel(usercount=ueCount)
snrCreater = SNRCreater()


ueAVGSNRS = []

#UE의 댓수 만큼 반복
for j in range(0,ueCount):
    #UE의 평균 snr 10~30 선정
    ueAVGSNRS.append(random.randrange(10,30))
        
q_model.ResetEpsilon()
throuput = []

episode = 10000
step = 1000000

#100번의 에피소드
for i in range(episode):
    totalthroughputs = 0
    #10000번의 스탭
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
    
        
        
        
        
        
        