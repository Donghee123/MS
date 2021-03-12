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
 
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.l1 = nn.Linear(4,100)
        self.l2 = nn.Linear(100,400)
        self.l3 = nn.Linear(400,100)
        self.l4 = nn.Linear(100,4)
        
    def forward(self, x):
        
        x = x.float()
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        h4 = self.l4(h3)
        return F.log_softmax(h4, dim=1)
    

class Dataset(data_utils.Dataset):
   
    def __init__(self, X, y):
        self.X = X
        self.y = y
   
    def __getitem__(self, idx):
        return {'X': self.X[idx], 'y': self.y[idx]}
   
    def __len__(self):
        return len(self.X)

seed = 1

X_features = ["UE0", "UE1", "UE2", "UE3"]
y_features = ["SelUE0", "SelUE1", "SelUE2", "SelUE3"]

batch_size = 1

trndataset = pd.read_csv('./dataset/train/batch0/dataset.csv')

trn_X_pd, trn_y_pd = trndataset[X_features], trndataset[y_features ]

trn_X = torch.from_numpy(trn_X_pd.astype(float).to_numpy())
trn_y = torch.from_numpy(trn_y_pd.astype(float).to_numpy())

trn = data_utils.TensorDataset(trn_X, trn_y)
trn_loader = data_utils.DataLoader(trn, batch_size=batch_size, shuffle=True)


testdataset = pd.read_csv('./dataset/test/batch0/dataset.csv')

test_X_pd, test_y_pd = testdataset[X_features], trndataset[y_features ]

test_X = torch.from_numpy(test_X_pd.astype(float).to_numpy())
test_y = torch.from_numpy(test_y_pd.astype(float).to_numpy())

test = data_utils.TensorDataset(test_X, test_y)
test_loader = data_utils.DataLoader(test, batch_size=batch_size, shuffle=True)

torch.manual_seed(seed)
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=0.001,momentum=0.5)
aryofLoss = []

def test(log_interval, model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            
            output = model(data)
            print(output)
            print(target)
            test_loss += criterion(output, torch.max(target, 1)[1], reduction='sum').item() 
            #pred = output.argmax(dim=1, keepdim=True)
            #correct += pred.eq(target.view_as(pred)).item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format
          (test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
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
            print(lossValue)
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
            
log_interval = 10
           

train(100)
test(log_interval, model, test_loader)
torch.save(model, './model.pt')    

"""
########################################################################
# 결과가 괜찮아보이네요.
#
# 그럼 전체 데이터셋에 대해서는 어떻게 동작하는지 보겠습니다.

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

########################################################################
# (10가지 분류 중에 하나를 무작위로) 찍었을 때의 정확도인 10% 보다는 나아보입니다.
# 신경망이 뭔가 배우긴 한 것 같네요.
#
# 그럼 어떤 것들을 더 잘 분류하고, 어떤 것들을 더 못했는지 알아보겠습니다:

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
"""