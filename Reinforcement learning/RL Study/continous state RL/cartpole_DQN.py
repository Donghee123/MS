"""
Created on Tue May 18 10:04:15 2021

@author: Handonghee
@Discription : 
    subject  : Cartpole tutorial
    method   : Deep Q Network
             
"""
import random
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

import gym
import numpy as np
import matplotlib.pyplot as plt
from JSAnimation.IPython_display import display_animation
from matplotlib import animation
from IPython.display import display
from collections import namedtuple

BATCH_SIZE = 32
CAPACITY = 10000


Transition = namedtuple('Transition', ('state','action','next_state', 'reward'))

def display_frames_as_gif(frames):
    plt.figure(figsize=(frames[0].shape[1]/72.0, frames[0].shape[0]/72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')
    
    def animate(i):
        patch.set_date(frames[i])
    
    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    
    anim.save('move_cartpole.mp4')
    display(display_animation(anim, default_mode='loop'))

class ReplayMemory:
    def __init__(self,CAPACITY):
        self.capacity = CAPACITY
        self.memory=[]
        self.index = 0
    
    def push(self,state,action,state_next,reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        
        self.memory[self.index] = Transition(state,action, state_next, reward)
        
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


        

class Brain:
    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions
        self.memory = ReplayMemory(CAPACITY)
        
        self.model = nn.Sequential()
        self.model.add_module('fc1', nn.Linear(num_states, 32))
        self.model.add_module('relu1', nn.ReLU())
        self.model.add_module('fc2', nn.Linear(32,32))
        self.model.add_module('relu2', nn.ReLU())
        self.model.add_module('fc3', nn.Linear(32,num_actions))
        
        print(self.model)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
    
    def replay(self):
        if len(self.memory) < BATCH_SIZE: 
            return
        
        transitions = self.memory.sample(BATCH_SIZE)
        
        batch = Transition(*zip(*transitions))
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        
        self.model.eval()
        
        state_action_values = self.model(state_batch).gather(1,action_batch)
        
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))
    
        next_state_values = torch.zeros(BATCH_SIZE)
        
        next_state_values[non_final_mask] = self.model(non_final_next_states).max(1)[0].detach()
        
        expected_state_action_values = reward_batch + GAMMA * next_state_values
        
        self.model.train()
        
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
        
       
    def decide_action(self, state, episode):      
        epsilon = 0.5 * (1 / (episode + 1))
        
        if epsilon <= np.random.uniform(0,1):
            self.model.eval()
            with torch.no_grad():
                action = self.model(state).max(1)[1].view(1,1)
        else:
            action = torch.LongTensor(
                [[random.randrange(self.num_actions)]])
        
        return action
            
             
class Agent:
    def __init__(self, num_states, num_actions):
        self.brain = Brain(num_states, num_actions)
        
    def update_q_function(self):
        self.brain.replay()
    
    def get_action(self, state, episode):
        action =self.brain.decide_action(state, episode)
        return action

    def momorize(self, state, action, state_next, reward):
        self.brain.memory.push(state, action, state_next, reward)
        
class Environment:
    def __init__(self):
        self.env = gym.make(ENV)
        num_states = self.env.observation_space.shape[0]
        num_actions = self.env.action_space.n
        self.agent = Agent(num_states, num_actions)
        
    def close(self):
        self.env.close()
        
    def run(self):
        episode_10_list = np.zeros(10)
        
        complete_episodes = 0        
        is_episode_final = False
        
        
        for episode in range(NUM_EPISODES):
            observation = self.env.reset()
            
            #상태 저장
            state = observation
            #numpy를 파이토치 텐서로 변환
            state = torch.from_numpy(state).type(torch.FloatTensor)
            
            #size 4를 size 1 * 4로 변환            
            state = torch.unsqueeze(state,0)
            
            for step in range(MAX_STEPS):
                
                #화면 보여주기
                self.env.render()
                    
                #파이토치 텐서로 변환 시킨 state와 episde 입력, action을 DNN 모델에서 나온 결과중 max 값
                action = self.agent.get_action(state, episode)
                
                #현재 action으로  1step 수행후 결과 리턴
                observation_next, _, done, _ = self.env.step(action.item())
                
                if done:
                    #다음 상태가 없으므로 None 지정
                    state_next = None
                    
                    #최근 10 에피소드에서 버틴 단계 수를 리스트에 저장
                    episode_10_list = np.hstack((episode_10_list[1:], step + 1))
                    
                    
                    if step < MAX_STEPS - 5:
                        reward = torch.FloatTensor([-1.0])
                        complete_episodes = 0
                    else:
                        reward = torch.FloatTensor([1.0])
                        complete_episodes += 1
                else:
                    reward = torch.FloatTensor([0.0])
                    state_next = observation_next
                    state_next = torch.from_numpy(state_next).type(torch.FloatTensor)
                    state_next = torch.unsqueeze(state_next,0)
                
                #메모리에 경험을 저장
                self.agent.momorize(state, action, state_next, reward)
                
                #Experience Replay로 Q함수를 수정
                self.agent.update_q_function()
                
                state = state_next
                
                
                if done:
                    print('{0} Episode : Finished after {1} time steps : 최근 10 에피소드의 평균 단계 수 = {2}'.format(episode, step+1, episode_10_list.mean()))
                    break
            
            #최종 결정한 에피소드라면 끝내기
            if is_episode_final is True:
                break
            
            if complete_episodes >= 10:
                print('10 에피소드 연속 성공')
                is_episode_final = True


#시뮬레이션 상수 값 정의
ENV = 'CartPole-v0'
GAMMA = 0.99
MAX_STEPS = 200
NUM_EPISODES = 500

cartpole_env = Environment()
cartpole_env.run()
cartpole_env.close()