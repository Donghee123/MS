# -*- coding: utf-8 -*-
"""
DNN based basestation scheduling model(1/2)

Chnnel model : Releigh fading
Make Dataset
[UE0, UE1, UE2, UE3, UE4, UE5, UE7, UE8 ,SelFirst, SelSecond]
"""
import random
import math
import numpy
import csv
import os

#File 유틸 함수들    
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
     
def MakeCSVFile(strFolderPath, strFilePath, aryOfDatas):
    strTotalPath = "%s\%s" % (strFolderPath,strFilePath)
    
    f = open(strTotalPath,'w', newline='')
    wr = csv.writer(f)
    wr.writerow(["UE0", "UE1", "UE2", "UE3","UE4", "UE5", "UE6", "UE7", "SelFirst", "SelSecond"])
    for i in range(0,len(aryOfDatas)):
        wr.writerow(aryOfDatas[i])
    
    f.close()
    
    
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

#만들고자 하는 batch 사이즈
nCreateBatchSize = 1

#8개의 UE
nUECount = 8

#10000개의 데이터 수집
nTestCount = 100000

#레일리페이딩 기반 랜덤 SNR 생성 Class 
snrCreater = SNRCreater()

#원하는 Batch 사이즈 만큼 반복
for nBatchCount in range(0,nCreateBatchSize):
    #모든 UE의 SNR 정보를 저장하는 컨테이너
    AryOfUserEquipmentsSnr = []
    #테스트 횟수 만큼 반복
    for i in range(0,nTestCount):
        AryOfUserEquipmentSnr = []
        
        
        #UE의 댓수 만큼 반복
        for j in range(0,nUECount):
            #UE의 평균 snr 10~30 선정
            favgSNR = random.randrange(10,30)
            AryOfUserEquipmentSnr.append(snrCreater.GetReleighSNR(favgSNR))         
        
        AryOfUserEquipmentsSnr.append(AryOfUserEquipmentSnr)
        
    AryOfMaxSNR = []
    
    #UE의 댓수 만큼 반복
    for i in range(0,nTestCount):
        #테스트 횟수 만큼 반복
        
        nMaxIndex = 0
        nSecondMaxIndex = 0
        AryOfSelectedUE = []
        
        #1등 인덱스 찾기
        fMaxValue = 0.0
        for j in range(0,nUECount):
            if fMaxValue < AryOfUserEquipmentsSnr[i][j]:
                fMaxValue = AryOfUserEquipmentsSnr[i][j]
                nMaxIndex = j
                
        #2등 인덱스 찾기
        fMaxValue = 0.0
        for j in range(0,nUECount): 
            if j == nMaxIndex:
                continue
                    
            if fMaxValue < AryOfUserEquipmentsSnr[i][j]:
                fMaxValue = AryOfUserEquipmentsSnr[i][j]
                nSecondMaxIndex = j
        
        AryOfUserEquipmentsSnr[i].append(nMaxIndex)
        AryOfUserEquipmentsSnr[i].append(nSecondMaxIndex)
         
                           
        AryOfMaxSNR.append(AryOfSelectedUE)
        
    strFolerPath = 'batch' + str(nBatchCount)   
    createFolder(strFolerPath)
   
    MakeCSVFile(strFolerPath, "DataSet.csv", AryOfUserEquipmentsSnr)
    
  

        
    
    