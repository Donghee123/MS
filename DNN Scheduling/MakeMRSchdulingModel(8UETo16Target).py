# -*- coding: utf-8 -*-
"""
DNN based basestation scheduling model
excute train
excute test
save model
분류기(Classifier) 학습하기
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable 
import torch.utils.data as data_utils 
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random
import math
import numpy
import csv
import os

#Neual Netrok nn.Module 상속 
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.l1 = nn.Linear(8,16)
        self.l2 = nn.Linear(16,32)
        self.l3 = nn.Linear(32,64)
        self.l4 = nn.Linear(64,128)
        self.l5 = nn.Linear(128,64)
        self.l6 = nn.Linear(64,32)
        self.l7 = nn.Linear(32,16)
        
    def forward(self, x):
        
        x = x.float()
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))   
        h4 = F.relu(self.l4(h3))   
        h5 = F.relu(self.l5(h4))   
        h6 = F.relu(self.l6(h5))   
        h7 = self.l7(h6)
        
        return F.log_softmax(h7, dim = 1)   

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
    
#테스트 함수    
def test(log_interval, model, test_loader):
    
    #평가 모드로 전환
    model.eval()
    
    #Test loss 수집용
    test_loss = 0
    
    #correct rate 수집용
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            
            #추론 시작
            output = model(data)
           
            #추론 결과 Loss 취득
            Tempoutput1 = output.clone()
            Tempoutput2 = output.clone()
        
            Tempoutput1[0][8:16] = 0
            Tempoutput2[0][0:8] = 0
        
            Temptarget1 = target.clone()
            Temptarget2 = target.clone()
        
            Temptarget1[0][8:16] = 0
            Temptarget2[0][0:8] = 0
        
            #오차 output, labels 
            test_loss = criterion(Tempoutput1, torch.max(Temptarget1, 1)[1])
            test_loss += criterion(Tempoutput2, torch.max(Temptarget2, 1)[1])    
            
            CompareOutput1 = Tempoutput1[0][0:8]
            CompareOutput2 = Tempoutput2[0][8:16]
            
            #추론 결과 직접 비교
            if torch.tensor([torch.max(CompareOutput1,0)[1]]) == torch.max(Temptarget1,1)[1] and torch.tensor([torch.max(CompareOutput2,0)[1]]) == (torch.max(Temptarget2,1)[1] - 8):
                correct += 1

    
    #추론 결과 평균 수집
    test_loss /= len(test_loader.dataset)

    #추론 결과 보여줌
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format
          (test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    artOfTempInfo = []
    artOfTempInfo.append(test_loss.item())
    artOfTempInfo.append(correct)
    artOfTempInfo.append(100. * correct / len(test_loader.dataset))
    aryofModelInfo.append(artOfTempInfo)
   

#학습 함수    
def train(epochcount, train_loader, validation_loader, nShowInterval):
    
    #model을 train 모드로 변경   
    running_loss = 0.0        
        #i : trn_loader index, data : set <inputs, labes>         
        
       
    for i, data in enumerate(train_loader):
        inputs, labels = data  
        model.train(True)
                                  
        #초기화
        optimizer.zero_grad()
                    
        #model 유추 시작
        output = model(inputs)
        
        #추론 결과 Loss 취득
        
        Tempoutput1 = output.clone()
        Tempoutput2 = output.clone()
        
        Tempoutput1[0][8:16] = 0
        Tempoutput2[0][0:8] = 0
        
        TempLabel1 = labels.clone()
        TempLabel2 = labels.clone()
        
        TempLabel1[0][8:16] = 0
        TempLabel2[0][0:8] = 0
        
        #오차 output, labels 
        test_loss = criterion(Tempoutput1, torch.max(TempLabel1, 1)[1])
        test_loss += criterion(Tempoutput2, torch.max(TempLabel2, 1)[1])    
               
        #loss = criterion(output, labels.float())
                
        #오차 역전파
        test_loss.backward()
        optimizer.step()
                                   
        #오차 값 표기
        lossValue = test_loss.item()
        running_loss += lossValue
                
        #nShowInterval번 순회시 누적 train loss 상태 보여주기, validation loss 상태 보여주기
        if i % nShowInterval == (nShowInterval - 1):    # print every 2000 mini-batches
            artoftempLoss = []
            print('[%d, %5d] loss: %.3f' %(epochcount + 1, i + 1, running_loss / nShowInterval))
            artoftempLoss.append(running_loss/nShowInterval)
            aryofLoss.append(artoftempLoss)
            running_loss = 0.0
            validationRunning_loss= 0.0
            
            #2000 trainning당 한번 validation 체크
            for valindex, valdata in enumerate(validation_loader):
                validationinputs, validationlabels = valdata 
                model.train(False)
                
                validationoutput = model(validationinputs) 
                
                valTempoutput1 = validationoutput.clone()
                valTempoutput2 = validationoutput.clone()
        
                valTempoutput1[0][8:16] = 0
                valTempoutput2[0][0:8] = 0
        
                valTempLabel1 = validationlabels.clone()
                valTempLabel2 = validationlabels.clone()
        
                valTempLabel1[0][8:16] = 0
                valTempLabel2[0][0:8] = 0
        
                #오차 output, labels 
                validationloss = criterion(valTempoutput1, torch.max(valTempLabel1, 1)[1])
                validationloss += criterion(valTempoutput2, torch.max(valTempLabel2, 1)[1])    
                           
                validationlossValue = validationloss.item()
                validationRunning_loss += validationlossValue
            
            validationTempLoss = []
            validationSize = len(data_loaders['val'])
            validationRunning_loss /= validationSize
            validationTempLoss.append(validationRunning_loss)
            aryofValidationLoss.append(validationTempLoss)
            print('Validationloss: %.3f' %(validationRunning_loss))
               
    
"""
Start Pandas API를 이용한 엑셀 데이터 취득 구간
"""
#100000개 데이터 수집
trndataset = pd.read_csv('./dataset/train/batch0/dataset.csv')
"""
End Pandas API를 이용한 엑셀 데이터 취득 구간
"""


"""
Start 훈련 및 테스트용 데이터 분할 구간
"""
seed = 1

#epochin
epoch = 50
#해당 각 데이터 범주 정의
X_features = ["UE0", "UE1", "UE2", "UE3","UE4", "UE5", "UE6", "UE7"]
y_features = ["1_SelUE0", "1_SelUE1", "1_SelUE2", "1_SelUE3", "1_SelUE4", "1_SelUE5", "1_SelUE6", "1_SelUE7", 
                 "2_SelUE0", "2_SelUE1", "2_SelUE2", "2_SelUE3", "2_SelUE4", "2_SelUE5", "2_SelUE6", "2_SelUE7"]

#batch size = 1
batch_size = 1

#범주 별로 데이터 취득 X : 입력, Y : 출력
trn_X_pd, trn_y_pd = trndataset[X_features], trndataset[y_features ]

#pandas.core.frame.DataFrame -> numpy -> torch 데이터형으로 변경
trn_X = torch.from_numpy(trn_X_pd.astype(float).to_numpy())
trn_y = torch.from_numpy(trn_y_pd.astype(float).to_numpy())

#torch DataSet(input, output) 세트로 변경
trn = data_utils.TensorDataset(trn_X, trn_y)


#데이터셋 중 훈련 : 70%, 검증 : 30% 사용
trainsetSize = int(70 * len(trn) / 100)
valisetSize = int(10 * len(trn) / 100)
testsetSize = len(trn) - (trainsetSize + valisetSize) 
#testsetSize = len(trn) - trainsetSize - valisetSize

trn_set, val_set, test_set = torch.utils.data.random_split(trn, [trainsetSize, valisetSize, testsetSize])

#훈련용 데이터 로더
trn_loader = data_utils.DataLoader(trn_set, batch_size=batch_size, shuffle=True)
#검증용 데이터 로더
val_loader = data_utils.DataLoader(val_set, batch_size=batch_size, shuffle=True)
#테스트용 데이터 로더
test_loader = data_utils.DataLoader(test_set, batch_size=batch_size, shuffle=True)

#범주별 data_loader 핸들링
data_loaders = {"train": trn_loader, "val": val_loader, "test":test_loader}
"""
End 훈련 및 테스트용 데이터 분할 구간
"""


"""
start module 인스턴스 생성 후 학습 및 테스트 구간
"""
torch.manual_seed(seed)
model = Net()
#MES Loss
criterion = nn.CrossEntropyLoss()
#최적화 함수 SGD, 학습률 0.001, momentum 0.5
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)

aryofLoss = []
aryofModelInfo = []
aryofValidationLoss = []
log_interval = 1
           
for i in range(0,epoch):
    #훈련
    train(i, data_loaders['train'], data_loaders['val'],2000)
    #평가
    test(log_interval, model, data_loaders['test'])

#Save Model 

modelPath = "./model"
createFolder(modelPath)   
MakeCSVFile(modelPath, "ModelLossInfo.csv", ["Loss"], aryofLoss)
MakeCSVFile(modelPath, "ModelValidationLossInfo.csv", ["Validation Loss"], aryofValidationLoss)
MakeCSVFile(modelPath, "ModelSpec.csv",["Average Loss","Correct","Accracy"],aryofModelInfo)
torch.save(model, modelPath + '/model.pt')    

