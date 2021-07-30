# -*- coding: utf-8 -*-
"""
DNN based basestation scheduling model
excute train
excute test
save model
분류기(Classifier) 학습하기
"""
import pandas as pd

import csv
import os


    
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
     
def MakeCSVFile(strFolderPath, strFilePath, artofHeader, aryOfDatas):
    strTotalPath = "%s\%s" % (strFolderPath,strFilePath)
    
    f = open(strTotalPath,'w', newline='')
    wr = csv.writer(f)
    
    wr.writerow(artofHeader)
    
    for i in range(0,len(aryOfDatas)):
        wr.writerow(aryOfDatas[i])
    
    f.close()
    

"""
Start Pandas API를 이용한 엑셀 데이터 취득 구간
"""
#100000개 데이터 수집
trndataset = pd.read_csv('./ModelLossInfo_Total.csv')
"""
End Pandas API를 이용한 엑셀 데이터 취득 구간
"""


"""
Start 훈련 및 테스트용 데이터 분할 구간
"""
seed = 1

#epochin

#해당 각 데이터 범주 정의
X_features = ["Loss"]
y_features = ["train"]

#batch size = 1
batch_size = 1

#범주 별로 데이터 취득 X : 입력, Y : 출력
trn_X_pd, trn_y_pd = trndataset[X_features], trndataset[y_features ]

#pandas.core.frame.DataFrame -> numpy -> torch 데이터형으로 변경
LossData = trn_X_pd.astype(float).to_numpy()
TrainCountData = trn_y_pd.astype(float).to_numpy()

epochLoss = []

for i in range(len(TrainCountData)):
    if TrainCountData[i][0] == 70000.0:
        temp = []
        temp.append(LossData[i][0])
        epochLoss.append(temp)
        
MakeCSVFile('./', "EpochLoss.csv", ["Loss"], epochLoss)
#Save Model 



#MakeCSVFile(modelPath, "ModelLossInfo.csv", ["Loss"], aryofLoss)
#MakeCSVFile(modelPath, "ModelValidationLossInfo.csv", ["Validation Loss"], aryofValidationLoss)
#MakeCSVFile(modelPath, "ModelSpec.csv",["Average Loss","Correct","Accracy"],aryofModelInfo)
#torch.save(model, modelPath + '/model.pt')    

