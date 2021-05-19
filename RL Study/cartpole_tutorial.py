# -*- coding: utf-8 -*-
"""
Created on Tue May 18 10:04:15 2021

@author: Handonghee
"""
import gym
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import animation
from IPython.display import display


#frames에 매번 영상 저장
frames = []

#CartPole 환경 구성
env = gym.make('CartPole-v0')

#환경 리셋
observation = env.reset()

while True:
    #frames.append()
    frames.append(env.render())
    action = np.random.choice(2) # 0(왼쪽으로), 1(오른쪽으로)
    observation, reward, done, info = env.step(action) # action 실행
    