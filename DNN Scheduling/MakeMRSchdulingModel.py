# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 21:50:10 2021

@author: Handonghee
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils 

from torchvision import transforms

import numpy as np
import pandas as pd

"""
def train(epoch):
    model.train()
    
    for data, targets in loader_train:
        optimizer.zero_grad()
        outputs = model(data)
        loss= loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
    
    print("epoch{}:완료\n".format(epoch))

def test():
    model.eval()
    correct = 0
    
    with torch.no_grad():
        
        for data, targets in loader_test:
            outputs = model(data)
            testdata,predicted = torch.max(outputs.data, 1)
            targetdata,targetpredicted = torch.max(targets.data, 1)
            
            correct = 0
            for i in range(len(predicted)):
                if predicted[i] == targetpredicted[i]:
                    correct += 1
            
            
    data_num = len(loader_test.dataset)
    print('\n테스트 데이터에서 예측 정확도' + str(correct) + '/' + str(data_num) + ' ' + str(100.*correct/data_num))
    

    
inputLayerdata = np.loadtxt("dataset/train/batch0/inputLayer.csv", delimiter=",", dtype=np.float32)
outputLayerdata = np.loadtxt("dataset/train/batch0/outputLayer.csv", delimiter=",", dtype=np.float32)
X_train, X_test, y_train, y_test = train_test_split(inputLayerdata,outputLayerdata, test_size=1/7, random_state=0)

X_train = torch.Tensor(X_train)
X_test = torch.Tensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

ds_train = TensorDataset(X_train,y_train)
ds_test = TensorDataset(X_test, y_test)

loader_train = DataLoader(ds_train, batch_size=10000, shuffle=True)
loader_test = DataLoader(ds_test, batch_size=10000, shuffle=False)

#############신경망 구성##################
model =nn.Sequential()
model.add_module('fc1', nn.Linear(4,100))
model.add_module('relu1', nn.ReLU())
model.add_module('fc2', nn.Linear(100,50))
model.add_module('relu2', nn.ReLU())
model.add_module('fc3', nn.Linear(50,4))

#############오차함수#####################
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.01)

#test()
for epoch in range(3):
    train(epoch)

test()
"""



class Dataset(data_utils.Dataset):
   
    def __init__(self, X, y):
        self.X = X
        self.y = y
   
    def __getitem__(self, idx):
        return {'X': self.X[idx], 'y': self.y[idx]}
   
    def __len__(self):
        return len(self.X)
    
class MLPRegressor(nn.Module):
    
    def __init__(self,X_features):
        super(MLPRegressor, self).__init__()
        h1 = nn.Linear(len(X_features), 50)
        h2 = nn.Linear(50, 35)
        h3 = nn.Linear(35, 1)
        self.hidden = nn.Sequential(
            h1,
            nn.Tanh(),
            h2,
            nn.Tanh(),
            h3,
        )
        
    def forward(self, x):
        x = x.float()
        o = self.hidden(x)
        return o.view(-1)
    
batch_size = 64

trninputdataset = pd.read_csv('./dataset/train/batch0/dataset.csv')
vrninputdataset = pd.read_csv('./dataset/var/batch0/dataset.csv')
testinputdataset = pd.read_csv('./dataset/test/batch0/dataset.csv')


X_features = ["UE0", "UE1", "UE2", "UE3"]
y_feature = ["SelectedUE"]

trn_X_pd, trn_y_pd = trninputdataset[X_features], trninputdataset[y_feature]
val_X_pd, val_y_pd = vrninputdataset[X_features], vrninputdataset[y_feature]
test_X_pd, test_y_pd = vrninputdataset[X_features], vrninputdataset[y_feature]

trn_X = torch.from_numpy(trn_X_pd.astype(float).to_numpy())
trn_y = torch.from_numpy(trn_y_pd.astype(float).to_numpy())

val_X = torch.from_numpy(val_X_pd.astype(float).to_numpy())
val_y = torch.from_numpy(val_y_pd.astype(float).to_numpy())

test_x = torch.from_numpy(test_X_pd.astype(float).to_numpy())
test_y = torch.from_numpy(test_y_pd.astype(float).to_numpy())

trn = data_utils.TensorDataset(trn_X, trn_y)
trn_loader = data_utils.DataLoader(trn, batch_size=batch_size, shuffle=True)

val = data_utils.TensorDataset(val_X, val_y)
val_loader = data_utils.DataLoader(val, batch_size=batch_size, shuffle=False)

test = data_utils.TensorDataset(test_x, test_y)
test_loader = data_utils.DataLoader(test, batch_size=batch_size, shuffle=False)

trn = Dataset(trn_X, trn_y)
trn_loader = data_utils.DataLoader(trn, batch_size=batch_size, shuffle=True)

val = Dataset(val_X, val_y)
val_loader = data_utils.DataLoader(val, batch_size=batch_size, shuffle=False)

test = Dataset(test_x, test_y)
test_loader = data_utils.DataLoader(test, batch_size=batch_size, shuffle=False)

model = MLPRegressor(X_features)
criterion = nn.MSELoss()
learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
num_epochs = 10
num_batches = len(trn_loader)

trn_loss_list = []
val_loss_list = []

for epoch in range(num_epochs):
    trn_loss_summary = 0.0
    for i, trn in enumerate(trn_loader):
        trn_X, trn_y = trn['X'], trn['y']
        
        optimizer.zero_grad()
        trn_pred = model(trn_X.float())
        trn_loss = criterion(trn_pred, trn_y.float())
        trn_loss.backward()
        optimizer.step()
        
        trn_loss_summary += trn_loss
        
        if (i+1) % 15 == 0:
            with torch.no_grad():
                val_loss_summary = 0.0
                for j, val in enumerate(val_loader):
                    val_X, val_y = val['X'], val['y']
                   
                    val_pred = model(val_X)
                    val_loss = criterion(val_pred, val_y)
                    val_loss_summary += val_loss
                
            print("epoch: {}/{} | step: {}/{} | trn_loss: {:.4f} | val_loss: {:.4f}".format(
                epoch + 1, num_epochs, i+1, num_batches, (trn_loss_summary/15)**(1/2), (val_loss_summary/len(val_loader))**(1/2)
            ))
                
            trn_loss_list.append((trn_loss_summary/15)**(1/2))
            val_loss_list.append((val_loss_summary/len(val_loader))**(1/2))
            trn_loss_summary = 0.0
        
print("finish Training")


model.eval()
correct = 0
    
with torch.no_grad():
        
    for data, targets in test_loader:
       
        
        print(targets.data)
               
                
    data_num = len(test_loader.dataset)
    print('\n테스트 데이터에서 예측 정확도' + str(correct) + '/' + str(data_num) + ' ' + str(100.*correct/data_num))
        


