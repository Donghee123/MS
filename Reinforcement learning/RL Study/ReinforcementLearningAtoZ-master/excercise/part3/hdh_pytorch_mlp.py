# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 10:48:52 2021

@author: DH
"""
#system package를 가져온 후 프로젝트의 루트를 추가함
import sys; sys.path.append('...') 

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from MLP import MultiLayerPerceptron as MLP
from linear_data import generate_samples

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer1 = nn.Linear(5,2)
        self.layer2 = nn.Linear(5,2)
        #학습중 업데이트가 되지만, 미분을 활용한 업데이트를 하지 않는 텐서들을 등록할 때
        #혹은 차후에 저장해야하는 텐서값 일때 ex) 입실론 정책의 입실론 값
        self.register_buffer('w',torch.zeros(5))

"""
Stochastic GD vs GD
"""
def train_model(model, opt, loader, epoch, criteria, xs, ys):
    
    losses = []
    
    for e in range(epoch):
        for x,y in loader:
        
            pred = model(x) #forward 함수를 통해 mlp 모델의 추정치(pred) 계산
            loss = criteria(pred,y) #pred와 target값 y의 MSE Loss 계산
        
            opt.zero_grad()
            loss.backward() #loss의 파라미터에 대한 편미분 계산
            opt.step() #경사 하강법을 이용해서 파라미터 계산
        
        #전체 데이터셋에 대한 오차 계산
        pred = model(xs)
        loss = criteria(pred,ys)
        loss = loss.detach().numpy() #torch.tensor를 numpy.array로 변환 detach홤수는 최적화에 도움됨
        losses.append(loss)
        #another option
        
        """
        with torch.no_grad():
            pred = mlp(xs)
            loss = criteria(pred,ys).numpy() 
            """    
        
    
    return losses

def run_minibatch_fullbatch(num_reps : int, n_sample : int, batch_size : int, epoch : int):
    criteria = torch.nn.MSELoss() #Loss 함수
    
    sgd_losses=[]
    gd_losses=[]
    
    for _ in range(num_reps):
        mlp = MLP(input_dim=1, output_dim=1, num_neurons=[64],hidden_act='Identity', out_act='Identity')
        opt_mlp = torch.optim.Adam(params=mlp.parameters(), lr=1e-3)
        
        mlp2 = MLP(input_dim=1, output_dim=1, num_neurons=[64],hidden_act='Identity', out_act='Identity')
        mlp2.load_state_dict(mlp.state_dict()) # mlp파라미터와 mlp2의 파라미터 테스트의 시작점을 통일하기 위해 load_state_dict 사용
        opt_mlp2 = torch.optim.Adam(params=mlp2.parameters(), lr=1e-3)
        
        xs, ys = generate_samples(n_sample)
        ds = torch.utils.data.TensorDataset(xs,ys)
        
        data_loader = torch.utils.data.DataLoader(ds, batch_size=batch_size)
        full_loader = torch.utils.data.DataLoader(ds, batch_size=n_sample)
        
        #SGD = Mini batch
        sgd_loss = train_model(mlp, opt_mlp, data_loader, epoch, criteria, xs, ys)
        sgd_losses.append(sgd_loss)
        
        #GD = Full batch
        gd_loss = train_model(mlp2, opt_mlp2, full_loader, epoch, criteria, xs, ys)
        gd_losses.append(gd_loss)
        
    sgd_losses = np.stack(sgd_losses)
    gd_losses = np.stack(gd_losses)
    
    return sgd_losses, gd_losses


mm = MyModel()

print('==============Chilren===============')
#등록한 레이어를 children으로 관리
for c in mm.children():
    print(c)
    
print('\n')

print('==============Parameters===============')
#mm 인스턴스의 모든 모듈들이 가지고있는 파라미터 출력
for p in mm.parameters():
    print(p)

print('\n')  

print('==============Use Device===============') 
#model.to('cpu')  #cpu쓰기
#model.to('cuda') #gpu쓰기
print('\n')  

print('==============Set train mode===============') 
#model.train() #학습모드 설정 self.train 값이 True가됨
#model.eval() #평가모드 설정 self.train 값이 False가됨
print('\n')  

print('==============MLP 만들기===============') 
#mlp 인스턴스 생성
mlp = MLP(input_dim=1, output_dim=1, num_neurons=[64,32],hidden_act='ReLU', out_act='Identity')

#MLP 뉴런 구조 보기
print(mlp) 

#MLP 모델의 파라미터 값들 보기
print(mlp.state_dict())

#mlp2 인스턴스 생성
mlp2 = MLP(input_dim=1, output_dim=1, num_neurons=[64,32],hidden_act='ReLU', out_act='Identity')

#mlp2 인스턴스에 mlp1의 파라미터 값들 복사 시키기 개꿀!!
mlp2.load_state_dict(mlp.state_dict())
print('\n')  

w = 1.0
b = 0.5
xs, ys = generate_samples(512,w=w,b=b)

ds = torch.utils.data.TensorDataset(xs,ys)
data_loader = torch.utils.data.DataLoader(ds, batch_size=64)
full_loader = torch.utils.data.DataLoader(ds, batch_size=512)

epoch = 64
opt = torch.optim.Adam(params=mlp.parameters(), lr=1e-3)
criteria = torch.nn.MSELoss()


sgd_losses, gd_losses = run_minibatch_fullbatch(50,128,32,30)

sgd_loss_mean = np.mean(sgd_losses, axis = 0)
gd_loss_mean = np.mean(gd_losses, axis = 0)

sgd_loss_std = np.std(sgd_losses, axis = -0)
gd_loss_std = np.std(gd_losses, axis = -0)

fig, ax = plt.subplots(1,1, figsize=(10,5))
ax.grid()
ax.fill_between(x=range(sgd_loss_mean.shape[0]), 
                y1 = sgd_loss_mean + sgd_loss_std,
                y2 = sgd_loss_mean - sgd_loss_std,
                alpha=0.3)
ax.plot(sgd_loss_mean, label="SGD")

ax.fill_between(x=range(gd_loss_mean.shape[0]), 
                y1 = gd_loss_mean + gd_loss_std,
                y2 = gd_loss_mean - gd_loss_std,
                alpha=0.3)
ax.plot(gd_loss_mean, label="GD")
ax.legend()

_ = ax.set_xlabel('epoch')
_ = ax.set_ylabel('loss')
_ = ax.set_title('SGD vs GD')


""" 
fig, ax = plt.subplots(1,1, figsize=(10,5))
ax.grid()
ax.plot(sgd_losses)
ax.set_xlabel('opt step')
ax.set_ylabel('loss')
ax.set_title('training loss curve')
"""     
        
        
