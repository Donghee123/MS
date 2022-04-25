# -*- coding: utf-8 -*-
"""
Created on Thu May 20 16:58:50 2021

@author: Handonghee
@Discription : 
    subject  : Q-learning based Scheduling
    method   : discrete state, Q-learning, Releigh channel model
               Reward = current sumSNR - current state Max current sumSNR 
"""
import numpy as np
import random
import math 
import matplotlib.pyplot as plt
import fileutill

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

#Data class
class NetworkData:
    def __init__(self, data, isDecibel = True):
        self.isDecibel = isDecibel
        self.storeValue = data
    
    #db값을 실수로
    def dB2real(self,fDBValue):
        return pow(10.0, fDBValue/10.0);

    #실수를 db값으로
    def	real2dB(self,fRealValue):
        return 10.0 * math.log10(fRealValue)
    
    #저장한 데이터 기반 Thoughput 반환
    def GetThoughput(self):
        if self.isDecibel == True:
            return math.log2(1.0 + self.dB2real(self.storeValue))
        else:
            return self.storeValue
        
    #저장한 데이터 기반 SNR db 반환
    def GetSNRdb(self):
        if self.isDecibel  == True:
            return self.storeValue 
        else:
            return self.real2dB(self.storeValue)
        
    
#User class
class UserEquipment:
    def __init__(self, avgSNR):
        self.avgSNR = NetworkData(avgSNR)       
        self.capacity = 0
        self.metrix = 0
        self.snrCreater = SNRCreater()
        
    def GetSNR(self):
        return self.snrCreater.GetReleighSNR(self.avgSNR.GetSNRdb())
    
#system class
class AnalysisSystem:
    def __init__(self):
        self.totalMeanThrouhput = 0
        self.Fairness = 0
   
def CreateSNRdbList(userEquipments):
    answerSNRs = []
    
    for userEquipment in userEquipments:
        answerSNRs.append(userEquipment.GetSNR())
        
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
        self.sumSNRStore = {} 
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
          
    def GetsumSNRValue(self, state):
        """        
        Parameters
        ----------
        state : list                        
            
        Returns
        -------
        None
        
        기능
        -------        
        해당 state에서 sumSNR값 불러오기
        """
        convertState = self.ConvertListToStringValue(state)
            
        if convertState in self.sumSNRStore:
            return self.sumSNRStore[convertState]
        else:
            return 0
        
    
    def SetsumSNRValue(self, state, sumSNR):
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
        해당 state에서 sumSNR값 저장
        """
        
        convertState = self.ConvertListToStringValue(state)
        self.sumSNRStore[convertState] = sumSNR
        
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
        
    def GetReward(self, aftersumSNR, preSumSNR):
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
        return (aftersumSNR - preSumSNR)
    
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
        
        firstSNR = state[selFirstSNR]
        secondSNR = state[selSecondSNR]
        sumSNR = firstSNR + secondSNR
        #Reward 계산
        throughput1, throughput2 = self.CalThrouhput(state[selFirstSNR],state[selSecondSNR])    
        
        self.throughputs =  throughput1 + throughput2
        
        #현재 상태에서 계산한 SNR 값 - 현재 상태에서 저장하고 있는 최대 SNR 값
        reward = self.GetReward(sumSNR, self.GetsumSNRValue(state))
        
       
        #새로 나온 throuput을값이 더 높다면 해당 상태의 throuput 저장
        if self.GetsumSNRValue(state) < sumSNR:
            self.SetsumSNRValue(state, sumSNR)   
            
        self.preFairness = self.fairness
        
        #Q-learning
        qValue[action] = qValue[action] + (self.alpha * (reward + self.gamma * nextqValue[np.argmax(nextqValue)] - qValue[action]))
        
        #update한 결과 저장
        self.SetQValue(state, qValue)
        
        
#simulation
ueCount = 4
updateEpsilon = 1.0001
alpha = 0.4
gamma = 0.05
q_model = Q_LearningModel(usercount=ueCount, updateEpsilon=updateEpsilon, alpha=alpha, gamma=gamma)

userEquipmentList = []
userThrouputs = []
systemThrouputs = []

userSaveThrouputs = []
systemSaveThrouput = []

analysisSystem = AnalysisSystem()
indicateAVGSNR = [2.0,4.0,6.0,8.0]


#UE의 댓수 만큼 반복
"""
# set snr random
for j in range(0,ueCount):
    #UE의 평균 snr 0~10 선정
    randomAvgSNR = random.randrange(0,10)
    userEquipmentList.append(UserEquipment(randomAvgSNR))
    userThrouputs.append([])
"""
  

# set snr indicate
for avgSNR in indicateAVGSNR:
    #UE의 평균 snr 0~10 선정    
    userEquipmentList.append(UserEquipment(avgSNR))
    userThrouputs.append([])
      
q_model.ResetEpsilon()


episode = 3000
step = 50000

#episode
for i in range(episode):
    
    #init data
    #init total data
    analysisSystem.totalMeanThrouhput = 0
    
    #init user data
    for index in range(len(userEquipmentList)):
        userEquipmentList[index].capacity = 0
        
    #step
    for j in range(step):
        
        ueOfSNRs = CreateSNRdbList(userEquipmentList)
        action = q_model.SelectAction(ueOfSNRs)
        nextUEOfSNRs = CreateSNRdbList(userEquipmentList)
        q_model.UpdateQValue(ueOfSNRs, action, nextUEOfSNRs)
        
        firstUserIndex, secondUserIndex = q_model.GetSelectUserPair(action)
        
        userEquipmentList[firstUserIndex].capacity +=  NetworkData(ueOfSNRs[firstUserIndex]).GetThoughput()
        userEquipmentList[secondUserIndex].capacity += NetworkData(ueOfSNRs[secondUserIndex]).GetThoughput()        
        analysisSystem.totalMeanThrouhput += q_model.throughputs
    
    tempuserData = []
    
    for index in range(len(userEquipmentList)):
        #누적 user throuput 평균
        userEquipmentList[index].capacity /= step
        #plot 전용 데이터 저장
        userThrouputs[index].append(userEquipmentList[index].capacity)
        #csv 저장용 user throuput데이터 
        tempuserData.append(userEquipmentList[index].capacity)
    
    #csv 저장용 user throuput데이터 
    userSaveThrouputs.append(tempuserData)
        
    #누적 throuput 평균
    analysisSystem.totalMeanThrouhput /= step
    
    #csv 저장용 system throuput데이터 
    systemSaveThrouput.append([analysisSystem.totalMeanThrouhput])
    
    #plot 전용 데이터 저장
    systemThrouputs.append(analysisSystem.totalMeanThrouhput)
    

#totalThrouput 
plt.subplot(211)  
plt.plot(systemThrouputs) 

#userThrouputs 
xstep = np.arange(0, len(userThrouputs[0]), 1)
plt.subplot(212)
plt.plot(xstep, userThrouputs[0],label="user0 : " + str(int(userEquipmentList[0].avgSNR.GetSNRdb())))
plt.plot(xstep, userThrouputs[1],label="user1 : " + str(int(userEquipmentList[1].avgSNR.GetSNRdb())))
plt.plot(xstep, userThrouputs[2],label="user2 : " + str(int(userEquipmentList[2].avgSNR.GetSNRdb())))
plt.plot(xstep, userThrouputs[3],label="user3 : " + str(int(userEquipmentList[3].avgSNR.GetSNRdb())))
plt.legend(loc='best')
plt.show()
  
  
simulationPath = "./simulation"
userlegendname = ['user' + str(i) for i in range(len(userThrouputs))]

fileutill.createFolder(simulationPath)   
fileutill.MakeCSVFile(simulationPath, "q_learning_systemThrouput.csv", ["systemThrouput"], systemSaveThrouput)
fileutill.MakeCSVFile(simulationPath, "q_learning_userThrouput.csv",userlegendname, userSaveThrouputs)
      
        
        
        
        