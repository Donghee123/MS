# -*- coding: utf-8 -*-
"""
DNN based basestation scheduling model
excute train
excute test
save model
분류기(Classifier) 학습하기
"""
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable 
import torchvision.transforms as transforms
import torch.utils.data as data_utils 
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#Neual Netrok nn.Module 상속 
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.l1 = nn.Linear(4,16)
        self.l2 = nn.Linear(16,20)
        self.l3 = nn.Linear(20,8)
        self.l4 = nn.Linear(8,4)
        
    def forward(self, x):
        
        x = x.float()
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        h4 = self.l4(h3)
        return F.log_softmax(h4, dim=1)

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
            test_loss += criterion(output, torch.max(target, 1)[1])
            
            #추론 결과 직접 비교
            if torch.max(target,1)[1] == torch.max(output,1)[1]:
                correct += 1

    
    #추론 결과 평균 수집
    test_loss /= len(test_loader.dataset)

    #추론 결과 보여줌
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format
          (test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

#학습 함수    
def train(epochcount):
    
    #model을 train 모드로 변경
    model.train()
    
    for epoch in range(epochcount): 
        running_loss = 0.0
        
        #i : trn_loader index, data : set <inputs, labes> 
        for i, data in enumerate(trn_loader,0):
            inputs, labels = data  
            
            #초기화
            optimizer.zero_grad()
            
            #model 유추 시작
            output = model(inputs)
           
            #오차 output, labels 
            loss = criterion(output,  torch.max(labels, 1)[1])
            
            #오차 역전파
            loss.backward()
            optimizer.step()
                           
            #오차 값 표기
            lossValue = loss.item()
            running_loss += lossValue
            
            #2000번 순회시 loss 상태 보여주기
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 2000))
                aryofLoss.append(running_loss/2000)
                running_loss = 0.0
            
    
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

#해당 각 데이터 범주 정의
X_features = ["UE0", "UE1", "UE2", "UE3"]
y_features = ["SelUE0", "SelUE1", "SelUE2", "SelUE3"]

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
valisetSize = len(trn) - trainsetSize

trn_set, val_set = torch.utils.data.random_split(trn, [trainsetSize, valisetSize])

#훈련용 데이터 로더
trn_loader = data_utils.DataLoader(trn_set, batch_size=batch_size, shuffle=True)
#검증용 데이터 로더
val_loader = data_utils.DataLoader(val_set, batch_size=batch_size, shuffle=True)
"""
End 훈련 및 테스트용 데이터 분할 구간
"""


"""
start module 인스턴스 생성 후 학습 및 테스트 구간
"""
torch.manual_seed(seed)
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=0.001,momentum=0.5)
aryofLoss = []


log_interval = 10
           

for i in range(0,5):
    #훈련
    train(1)
    #평가
    test(log_interval, model, val_loader)

#Save Model    
torch.save(model, './model.pt')    

