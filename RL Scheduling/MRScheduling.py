# -*- coding: utf-8 -*-
"""
Created on Thu May 20 16:58:50 2021

@author: Handonghee
@Discription : 
    subject  : MR Scheduling
    method   : Releigh channel model
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

def CreateSNRdbList(userEquipments):
    answerSNRs = []
    
    for userEquipment in userEquipments:
        answerSNRs.append(userEquipment.GetSNR())
        
    return answerSNRs
        
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
 
#MRScheduler
class MRScheduler:
                   
    def SelectAction(self,state):
        
        selectCount = 2
        selectedUser = []
        for x in range(selectCount):
            maxValue = -1000
            maxindex = -1
            for i in range(len(state)):
                
                if i in selectedUser:
                    continue
                
                if maxValue < state[i]:
                    maxValue = state[i]
                    maxindex = i
            
            selectedUser.append(maxindex)
            
            
        return selectedUser          
        
#simulation
ueCount = 4
mrScheduler = MRScheduler()

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
        
episode = 3000
step = 10000

  
#init data
#init total data

for i in range(episode):
    analysisSystem.totalMeanThrouhput = 0
       
    #init user data
    for index in range(len(userEquipmentList)):
        userEquipmentList[index].capacity = 0
            
    #step
    for j in range(step):
            
        ueOfSNRs = CreateSNRdbList(userEquipmentList)
        action = mrScheduler.SelectAction(ueOfSNRs)
            
        userEquipmentList[action[0]].capacity += NetworkData(ueOfSNRs[action[0]]).GetThoughput()
        userEquipmentList[action[1]].capacity += NetworkData(ueOfSNRs[action[1]]).GetThoughput()        
        analysisSystem.totalMeanThrouhput += NetworkData(ueOfSNRs[action[0]]).GetThoughput() + NetworkData(ueOfSNRs[action[1]]).GetThoughput()  
        
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
fileutill.MakeCSVFile(simulationPath, "mr_systemThrouput.csv", ["systemThrouput"], systemSaveThrouput)
fileutill.MakeCSVFile(simulationPath, "mr_userThrouput.csv",userlegendname, userSaveThrouputs)
      
        

