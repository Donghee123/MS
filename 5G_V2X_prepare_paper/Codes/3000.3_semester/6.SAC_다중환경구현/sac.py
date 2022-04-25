import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update
from model import GaussianPolicy, QNetwork, DeterministicPolicy
import numpy as np
from replay_memory import ReplayMemory
import random
import pandas as pd
import csv
import os


#File 유틸 함수들    
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
     
def MakeCSVFile(strFolderPath, strFilePath, aryOfHedaers, aryOfDatas):
    strTotalPath = "%s\%s" % (strFolderPath,strFilePath)
    
    f = open(strTotalPath,'w', newline='')
    wr = csv.writer(f)
    wr.writerow(aryOfHedaers)
    
    for i in range(0,len(aryOfDatas)):
        wr.writerow(aryOfDatas[i])
    
    f.close()
    
class SAC(object):
    def __init__(self, num_inputs, action_space, args, env):

        self.env = env
        self.train_step = args.train_step
        self.test_step = args.test_step
        self.num_vehicle = env.n_Veh
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.action_space = action_space
        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning
        self.args = args
        
        self.memory = ReplayMemory(args.replay_size, args.seed)
        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.device = 'cpu'
        
        print('test wiil be processing by ', self.device)
        self.RB_number = 20
        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)
        self.training = True
        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape[0]).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def find_nearest_value(self,array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]    
    
    def find_nearest_arg(self,array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return idx

    def GetRB_Power(self,powerMin,action, env):
        
        powerRange = 23.0 - powerMin
        
        select_maxValue  = np.max(action[0:20])
        candidateList_selRB = np.where(action[0:20] == select_maxValue)
        selectedRB_index = random.sample(list(candidateList_selRB[0]),k=1)[0]

        
        actionFromPolicy = (action[20] + 1.0) * (powerRange / 2) + powerMin 
        selectedPower_index = self.find_nearest_arg(env.V2V_power_dB_List, actionFromPolicy)
        
        return selectedRB_index, selectedPower_index

    def select_action(self, state, evaluate=False):

        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)

        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)

        return np.array(action.detach().cpu()[0])

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device)


        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None, ):
        
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = 'models/_sac_actor_{}_{}'.format(env_name, suffix)
        if critic_path is None:
            critic_path = 'models/_sac_critic_{}_{}'.format(env_name, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    # Load model parameters
    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))
    
    
    
    """
    Scaling 기능 추가 2021/10/25
    0~999 -> -100.0 ~ 123.0
    """
    def GenerateAction(self, action : float):
        
        selected_resourceBlock = 0
        selected_powerdB = -100.0
        
        onestep_PowerdB = 123.0/999.0
        
        if action < 0.0:
            selected_resourceBlock = 0
            selected_powerdB = 0.0
        elif action >= 20000:
            selected_resourceBlock = 19
            selected_powerdB = 123.0
        else:
            selected_resourceBlock = int(action / 1000)
            selected_powerdB = (action % 1000) * onestep_PowerdB
                            
        if selected_powerdB > 123.0:
            selected_powerdB = 123.0
                            
        selected_powerdB -= 100.0
        
        return selected_resourceBlock, selected_powerdB
    
    
    def ConvertToRealAction(self, action):
        
        fselect_maxValue  = np.max(action[0:20])
        listofCandidateList_selRB = np.where(action[0:20] == fselect_maxValue)
        nindex_selectedRB = random.sample(list(listofCandidateList_selRB[0]),k=1)[0]

        fPower = action[20]
        return nindex_selectedRB, fPower

    
