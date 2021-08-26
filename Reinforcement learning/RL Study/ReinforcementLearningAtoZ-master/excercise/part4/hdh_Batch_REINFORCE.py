# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 09:45:12 2021

@author: DH
배운 것
1. wandb 사용법 -> join이 안됨 추후에 사용법 다시 해보기
2. 배치 에피소딕 REINFORCE
"""
import sys; sys.path.append('..')

import wandb
import json

import gym
import torch
import matplotlib.pyplot as plt

from MLP import MultiLayerPerceptron as MLP
from PolicyGradient import REINFORCE
from common.train_utils import EMAMeter, to_tensor
from common.memory.episodic_memory import EpisodicMemory

env = gym.make('CartPole-v1')
s_dim = env.observation_space.shape[0]
a_dim = env.action_space.n

net = MLP(s_dim, a_dim)
agent = REINFORCE(net)

#EpisodicMemory는 sample(s,a,r,ns,done)을 저장 하고있음.
memory = EpisodicMemory(max_size=100, gamma=1.0)
ema = EMAMeter()

n_eps = 10000
update_every = 2
print_every = 50

#config = dict()
#config['n_eps'] = n_eps
#config['update_every'] = update_every

#wandb.init(project='my-first-wandb-project', config=config)

returns = []
for ep in range(n_eps):
    s = env.reset()
    cum_r = 0
    
    states = []
    actions = []
    rewards = []
    
    while True:
        s = to_tensor(s,size=(1,4))
        a = agent.get_action(s)
        ns, r, done, info = env.step(a.item())
        
        #preprocess data
        r = torch.ones(1,1) * r
        done = torch.ones(1,1) * done
        
        memory.push(s, a, r, torch.tensor(ns), done)
        
        s = ns
        cum_r += r
        if done:
            break
        
    ema.update(cum_r)
    returns.append(cum_r)
    
    if ep % print_every == 0:
        print("Episode {} || EMA: {}".format(ep, ema.s))
    
   
    if ep % update_every == 0:
        s, a, _, _, done, g = memory.get_samples()
        agent.update_episodes(s, a, g, use_norm=True)
        memory.reset()

plt.plot(returns)
plt.show()    
    #use wandb
    #log_dict = dict()
    #log_dict['cum_return'] = cum_r
    #log 저
    #wandb.log(log_dict)
    
#Save model and experiment configuration
#json_val = json.dumps(config)
#with open(join(wandb.run.dir, 'config.json'),'w') as f:
    #json.dump(json_val, f)


#torch.save(agent.state_dict(), join(wandb.run.dir,'model.pt'))

#close wandb session
#wandb.join()
