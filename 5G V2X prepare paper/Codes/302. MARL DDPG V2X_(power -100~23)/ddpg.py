
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from model import (Actor, Critic)
from memory import SequentialMemory
from random_process import OrnsteinUhlenbeckProcess
from util import *

# from ipdb import set_trace as debug

criterion = nn.MSELoss()

class DDPG(object):
    def __init__(self, nb_states, nb_actions, args, num_vehicle):
        
        if args.seed > 0:
            self.seed(args.seed)

        self.nb_states = nb_states
        self.nb_actions= nb_actions
        
        # Create Actor and Critic Network
        net_cfg = {
            'hidden1':args.hidden1, 
            'hidden2':args.hidden2, 
            'init_w':args.init_w
        }
        self.actor = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_target = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_optim  = Adam(self.actor.parameters(), lr=args.prate)

        self.critic = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_target = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_optim  = Adam(self.critic.parameters(), lr=args.rate)
        
        self.target_update_step = args.target_update_step
        self.update_count = 0
        
        hard_update(self.actor_target, self.actor) # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)
        
        self.num_vehicle = num_vehicle
        self.action_all_with_power = np.zeros([self.num_vehicle, 3, 2],dtype = 'float')   # this is actions that taken by V2V links with power
        self.action_all_with_power_training = np.zeros([20, 3, 2],dtype = 'float')   # this is actions that taken by V2V links with power
        
        #Create replay buffer
        self.memory = SequentialMemory(limit=args.rmsize, window_length=args.window_length)
        self.random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=args.ou_theta, mu=args.ou_mu, sigma=args.ou_sigma)

        # Hyper-parameters
        self.batch_size = args.bsize
        self.tau = args.tau
        self.discount = args.discount
        self.depsilon = 1.0 / args.epsilon

        # 
        self.epsilon = 1.0
        self.s_t = None # Most recent state
        self.a_t = None # Most recent action
        self.is_training = True
        
        USE_CUDA = torch.cuda.is_available()
        
        print('test will be processed by cuda : ', USE_CUDA)
        # 
        if USE_CUDA: self.cuda()

    def update_policy(self, isHardupdate = False):
        # Sample batch
        state_batch, action_batch, reward_batch, \
        next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)

        # Prepare for the target q batch
        next_q_values = self.critic_target([
            to_tensor(next_state_batch, volatile=True),
            self.actor_target(to_tensor(next_state_batch, volatile=True)),
        ])
        
        next_q_values.volatile=False
        
        target_q_batch = to_tensor(reward_batch) + \
            self.discount*to_tensor(terminal_batch.astype(np.float))*next_q_values

        # Critic update
        self.critic.zero_grad()

        q_batch = self.critic([ to_tensor(state_batch), to_tensor(action_batch) ])
        
        value_loss = criterion(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optim.step()

        # Actor update
        self.actor.zero_grad()

        policy_loss = -self.critic([
            to_tensor(state_batch),
            self.actor(to_tensor(state_batch))
        ])

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        self.update_count += 1 
        # Target update
        if isHardupdate == True and self.update_count % self.target_update_step == 0:
            print('hard update..')
            hard_update(self.actor_target, self.actor) # Make sure target is with the same weigh
            hard_update(self.critic_target, self.critic)
        else:
            print('soft update..')
            soft_update(self.actor_target, self.actor, self.tau)
            soft_update(self.critic_target, self.critic, self.tau)
        
        return value_loss, policy_loss

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def observe(self, r_t, s_t1, done):
        if self.is_training:
            self.memory.append(self.s_t, self.a_t, r_t, done)
            self.s_t = s_t1

    def random_action(self):
        action = np.zeros(2)
        
        action[0] = np.random.uniform(0.,3.99)
        action[1] = np.random.uniform(-100.,23.0)#적용 power -100~23 
        self.a_t = action
        return action

    def select_action(self, s_t, decay_epsilon=True):
        action = to_numpy(
            self.actor(to_tensor(np.array([s_t])))
        ).squeeze(0)
        action += self.is_training*max(self.epsilon, 0)*self.random_process.sample()
        
        action[0] = action[0] + 1
        action[1] = action[1] + 1        
        action[0] = action[0] * 1.999
        action[1] = (action[1] * 61.5) - 100
                        
        action[0] = np.clip(action[0], 0., 3.99)
        action[1] = np.clip(action[1], -100., 23.0) #적용 power -100~23 
        
        if decay_epsilon:
            self.epsilon -= self.depsilon
        
        self.a_t = action
        return action

    def reset(self, obs):
        self.s_t = obs
        self.random_process.reset_states()
     
    def reset_random_process(self):
        self.random_process.reset_states()
        
    def reset_state(self, obs):
        self.s_t = obs
        
    def load_weights(self, output):
        if output is None: return

        self.actor.load_state_dict(
            torch.load('{}/actor.pkl'.format(output))
        )

        self.critic.load_state_dict(
            torch.load('{}/critic.pkl'.format(output))
        )

    def load_weights(self, actor_path, critic_path):
        
        self.actor.load_state_dict(
            torch.load(actor_path)
        )

        self.critic.load_state_dict(
            torch.load(critic_path)
        )
        
    def save_model(self,output, performance):
        
        savePath = output
        
        saveActorFileName = savePath + '/actor_' + performance + '.pkl'
        saveCriticFileName = savePath + '/critic_' + performance + '.pkl'
        
        torch.save(
            self.actor.state_dict(),
            saveActorFileName
        )
        torch.save(
            self.critic.state_dict(),
            saveCriticFileName
        )

    def seed(self,s):
        torch.manual_seed(s)
        if USE_CUDA:
            torch.cuda.manual_seed(s)