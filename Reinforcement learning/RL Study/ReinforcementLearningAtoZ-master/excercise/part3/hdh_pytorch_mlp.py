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




