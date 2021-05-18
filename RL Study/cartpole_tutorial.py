# -*- coding: utf-8 -*-
"""
Created on Tue May 18 10:04:15 2021

@author: Handonghee
"""
import gym
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from JSAnimation.IPython_display import display_animation
from matplotlib import animation
from IPython.display import display

def display_frames_as_gif(frames):
    """
    Display a list of frames a gif, with controls
    """
    
    plt.figure(figsize=(frames[0].shape[1]/72.0,frames[0].shape[0]/72.0),dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')
    
    def animate(i):
        patch.set_data(frames[i])
    
    anim = animation.FuncAnimation(plt.gcf(),animate, frames=len(frames), internal=50)
    anim.save('movie_cartpole.mp4')#애니메이션 저장
    display(display_animation(anim, defaule_mode='loop'))

frames = []
env = gym.make('CartPole-v0')
observation = env.reset()

for step in range(0,200):
    #frames 리스트에 rgb칼라 이미지 추가
    frames.append(env.render(mode='rgb_array'))
    action = np.random.choice(2) # 0(왼쪽으로), 1(오른쪽으로)
    observation, reward, done, info = env,step(action) # action 실행
    