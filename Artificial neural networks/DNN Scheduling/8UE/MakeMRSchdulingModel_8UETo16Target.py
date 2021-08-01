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
        self.l4 = nn.Linear(64,32)
        self.l5 = nn.Linear(32,16)
        
    def forward(self, x):
        
        x = x.float()
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))   
        h4 = F.relu(self.l4(h3))   
        h5 = F.sigmoid(self.l5(h4))    
       
        return h5

#폴더 생성 유틸 함수    
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

#테스트 결과 저장 유틸 함수
def MakeCSVFile(strFolderPath, strFilePath, aryofHeader, aryOfDatas):
    strTotalPath = "%s\%s" % (strFolderPath,strFilePath)
    
    f = open(strTotalPath,'w', newline='')
    wr = csv.writer(f)
    
    wr.writerow(aryofHeader)
    
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
            
              
            #오차 output, labels 
            test_loss += criterion(output, target.float())
                       
            target1 = target[0][0:8]
            target2 = target[0][8:16]
            
            output1 = output[0][0:8]
            output2 = output[0][8:16]
            
            #추론 결과 직접 비교
            if torch.max(target1,0)[1] == torch.max(output1,0)[1] and torch.max(target2,0)[1] == torch.max(output2,0)[1]:
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
                
        #오차 output, labels 
        test_loss = criterion(output, labels.float())
              
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
        
                #오차 output, labels 
                validationloss = criterion(validationoutput, validationlabels.float())
                          
                validationlossValue = validationloss.item()
                validationRunning_loss += validationlossValue
            
            validationTempLoss = []
            validationSize = len(validation_loader)
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
Start cuda device
"""
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

"""
End cuda device
"""

"""
Start 훈련 및 테스트용 데이터 분할 구간
"""
seed = 1

#epochin
epoch = 500
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

trn_set, val_set, test_set = torch.utils.data.random_split(trn, [trainsetSize, valisetSize, testsetSize])

is_cudaallocated = True
#훈련용 데이터 로더
trn_loader = data_utils.DataLoader(trn_set, batch_size=batch_size, shuffle=True, pin_memory=is_cudaallocated)
#검증용 데이터 로더
val_loader = data_utils.DataLoader(val_set, batch_size=batch_size, shuffle=True, pin_memory=is_cudaallocated)
#테스트용 데이터 로더
test_loader = data_utils.DataLoader(test_set, batch_size=batch_size, shuffle=True, pin_memory=is_cudaallocated)

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
#MES Los
criterion = nn.BCELoss()
#criterion = nn.MSELoss()
#최적화 함수 SGD, 학습률 0.001, momentum 0.5
optimizer = optim.SGD(model.parameters(), lr=0.001)
#최적화 함수 ADAM, 학습률 0.001, momentum 0.5
#optimizer = optim.Adam(model.parameters(), lr=0.01)
aryofLoss = []
aryofModelInfo = []
aryofValidationLoss = []
log_interval = 1
           
for i in range(0,epoch):
    #훈련
    train(i, data_loaders['train'], data_loaders['val'],trainsetSize)
    #평가
    test(log_interval, model, data_loaders['test'])

"""
End module 인스턴스 생성 후 학습 및 테스트 구간
"""

"""
Start Save module 인스턴스 생성 후 학습 및 테스트 구간
"""
modelPath = "./model"
createFolder(modelPath)   
MakeCSVFile(modelPath, "ModelLossInfo.csv", ["Loss"], aryofLoss)
MakeCSVFile(modelPath, "ModelValidationLossInfo.csv", ["Validation Loss"], aryofValidationLoss)
MakeCSVFile(modelPath, "ModelSpec.csv",["Average Loss","Correct","Accracy"],aryofModelInfo)
torch.save(model, modelPath + '/model.pt')    
"""
End Save module 인스턴스 생성 후 학습 및 테스트 구간
"""
