# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

Author : HDH

파이토치로 선형회귀 모델 구현 y=ws+b
Optimizer : SGD, Adam 비교 해보기

"""

import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

def generate_sample(n_samples: int, w: float = 1.0, b: float = 0.5, x_range=[-1.0,1.0]):
    xs = np.random.uniform(low=x_range[0], high=x_range[1],size=n_samples)
    ys = xs+b
    
    xs = torch.tensor(xs).view(-1,1).float()
    ys = torch.tensor(ys).view(-1,1).float()
    return xs,ys

def run_sgd(n_steps: int = 1000, report_every: int = 100, verbose=True):
    lin_model = nn.Linear(in_features=1,out_features=1)
    opt = torch.optim.SGD(params=lin_model.parameters(),lr=0.01)
    sgd_losses = []
    
    for i in range(n_steps):
        ys_hat = lin_model(xs)
        loss = criteria(ys_hat, ys)
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        if i % report_every == 0:
            if verbose:
                print('\n')
                print('{}th update: {}'.format(i,loss))
                for p in lin_model.parameters():
                    print(p)
            sgd_losses.append(loss.log10().detach().numpy())
    return sgd_losses

def run_adam(n_steps: int = 1000, report_every: int = 100, verbose=True):
    
    lin_model = nn.Linear(in_features=1,out_features=1)
    opt = torch.optim.Adam(params=lin_model.parameters(),lr=0.01)
    sgd_losses = []
    
    for i in range(n_steps):
        ys_hat = lin_model(xs)
        loss = criteria(ys_hat, ys)
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        if i % report_every == 0:
            if verbose:
                print('\n')
                print('{}th update: {}'.format(i,loss))
                for p in lin_model.parameters():
                    print(p)
            sgd_losses.append(loss.log10().detach().numpy())
    return sgd_losses
    
#랭크1 / 사이즈1 
x = torch.ones(1) * 2

#편미분으로 변할 값 requires_grad에서 편미분을 한다는 것임
w = torch.ones(1, requires_grad=True)

y = w * x

#편미분 계산하기
#y의 결과에 대해서 requires_grad=True로 지정한 tensor를 편미분함
y.backward()

#위의 식에 따르면 y = 2*w 임 -> 이를 dy/dw 로 미분 하면 2가됨
"""
print(w.grad)
"""

"""
선형 회귀 시작  1
nn.Linear
y= Wx + b 계산 해보기
"""

#입력 dimension : 10, 출력 dimension : 1 
lin = nn.Linear(in_features=10, out_features=1)

"""
for p in lin.parameters():
    print(p)
    print(p.shape)
    print('\n')
"""

#weight 값 모두 1로 초기화
lin.weight.data = torch.ones_like(lin.weight.data)

#bias 값 5로 초기화
lin.bias.data = torch.ones_like(lin.bias.data) * 5.0

"""
for p in lin.parameters():
    print(p)
    print(p.shape)
    print('\n')
"""

#torch.ones(10)을 안쓴 이유는 파이토치는 앞전에 있는 데이터는 batch로 취급하기 때문임
#지금 x는 batch size : 1, vector size : 10임
x = torch.ones(1, 10)

# 15가 나옴 y = [1,1,...1] * [1,1,...1] + 5.0 이므로 
y_hat = lin(x)

"""
print(y_hat.shape)
print(y_hat)
"""

"""
선형 회귀 시작  2
선형 회귀 구현 하기
"""

w = 1.0
b = 0.5
xs, ys = generate_sample(30,w=w,b=b)

lin_model = nn.Linear(in_features=1,out_features=1)

"""
for p in lin_model.parameters():
    print(p)
    print(p.shape)
    print('\n')
""" 
ys_hat = lin_model(xs) #lin_model로 예측하기

#nn.MSELoss 클래스를 criteria라는 인스턴스로 생성 
criteria = nn.MSELoss()

#loss 값 구하기 파라미터 : 예측값, 타겟값
loss = criteria(ys_hat, ys)

#optimizer생성 : Stochastic Gradiant Descent 파라미터 : SGD를 수행할 파라미터, 학습률 넣기
opt = torch.optim.SGD(params=lin_model.parameters(), lr=0.01)

#backward를 계산하기전에 편미분 계산에 필요한 텐서들의 편미분값을 초기화함
opt.zero_grad()

"""
for p in lin_model.parameters():
    print(p)
    print(p.grad)
"""

#loss값을 편미분 시킴
loss.backward()

#편미분한 값을 optimizer 함수에 적용함
opt.step()



"""
for p in lin_model.parameters():
    print(p)
    print(p.grad)
"""

"""
sgdlosses = run_sgd()
adamlosses = run_adam()
"""

"""
Adam과 SGD의 성능 차이 비교
"""

sgd_losses = [run_sgd(verbose=False) for _ in range(50)]
sgd_losses = np.stack(sgd_losses)

#평균
sgd_loss_mean = np.mean(sgd_losses, axis = 0)
#표준편차
sgd_loss_std = np.std(sgd_losses, axis = -0)

adam_losses = [run_adam(verbose=False) for _ in range(50)]
adam_losses = np.stack(adam_losses)

#평균
adam_loss_mean = np.mean(adam_losses, axis = 0)
#표준편차
adam_loss_std = np.std(adam_losses, axis = -0)

fig, ax = plt.subplots(1,1, figsize=(10,5))
ax.grid()

#표준 편차를 적용하는 그래프
ax.fill_between(x=range(sgd_loss_mean.shape[0]),
                y1=sgd_loss_mean+sgd_loss_std,
                y2=sgd_loss_mean-sgd_loss_std,
                alpha=0.3)

#50번 loss의 평균 값
ax.plot(sgd_loss_mean, label='SGD')

#표준 편차를 적용하는 그래프
ax.fill_between(x=range(adam_loss_mean.shape[0]),
                y1=adam_loss_mean + adam_loss_std,
                y2=adam_loss_mean - adam_loss_std,
                alpha=0.3)

#50번 loss의 평균 값
ax.plot(adam_loss_mean, label='Adam')
ax.legend()





