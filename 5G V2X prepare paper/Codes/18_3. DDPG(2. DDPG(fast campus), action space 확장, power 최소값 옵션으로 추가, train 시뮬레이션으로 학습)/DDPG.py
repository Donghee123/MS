import numpy as np
import torch
import torch.nn as nn

from MLP import MultiLayerPerceptron as MLP


class OrnsteinUhlenbeckProcess:
    """
    OU process; The original implementation is provided by minimalRL.
    https://github.com/seungeunrho/minimalRL/blob/master/ddpg.py
    """

    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.1, 0.01, 0.1
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x


class Actor(nn.Module):

    def __init__(self, input_state, output_action, num_neurons):
        super(Actor, self).__init__()
        self.mlp = MLP(input_state, output_action,
                       num_neurons=num_neurons,
                       hidden_act='ReLU',
                       out_act='Identity')

    def forward(self, state):
        # Action space of Pendulum v0 is [-2.0, 2.0]
        # 행동공간을 -2 ~ 2로 제한 시킴.
        # 출력 레이어의 활성화 함수를 탄젠트하이퍼볼릭을 사용하는것도 괜찮 하지만 BP에서 어려움을 겪음
        # 그 대신 모델에서 나오는 값을 -2 ~ 2로 clamp 시킴.
        return self.mlp(state).clamp(-2.0, 2.0)


class Critic(nn.Module):

    def __init__(self, input_state, output_action, num_neurons):
        super(Critic, self).__init__()
        
        #sate network, action network의 출력을 합침.
        self.q_estimator = MLP(input_state + output_action, 1,
                               num_neurons=[num_neurons[0], num_neurons[1],num_neurons[2]],
                               hidden_act='ReLU',
                               out_act='Identity')

    def forward(self, s, a):
        #2개의 네트워크를 합침 -> emb
        #cat -> concatenate를 의미
        emb = torch.cat([s, a], dim=-1)
        #emb값을 q_estimator에 넣음.
        return self.q_estimator(emb)


class DDPG(nn.Module):

    def __init__(self,
                 n_Veh, critic: nn.Module,
                 critic_target: nn.Module,
                 actor: nn.Module,
                 actor_target: nn.Module,
                 lr_critic: float = 0.0005,
                 lr_actor: float = 0.001,
                 gamma: float = 0.99):
        """
        :param critic: critic network
        :param critic_target: critic network target
        :param actor: actor network
        :param actor_target: actor network target
        :param lr_critic: learning rate of critic
        :param lr_actor: learning rate of actor
        :param gamma:
        """

        super(DDPG, self).__init__()
        
        self.critic = critic #critic 네트워크
        self.actor = actor #actor 네트워크
        self.lr_critic = lr_critic #critic의 lr은 보통 다름
        self.lr_actor = lr_actor  #actor의 lr은 보통 다름
        self.gamma = gamma

        # setup optimizers Critic
        self.critic_opt = torch.optim.Adam(params=self.critic.parameters(),
                                           lr=lr_critic)
        # setup optimizers Actor
        self.actor_opt = torch.optim.Adam(params=self.actor.parameters(),
                                          lr=lr_actor)

        # setup target Critic networks
        critic_target.load_state_dict(critic.state_dict())
        self.critic_target = critic_target
        
        # setup target Actor networks
        actor_target.load_state_dict(actor.state_dict())
        self.actor_target = actor_target
        
        # Loss 함수는 SmoothL1Loss
        self.criteria = nn.SmoothL1Loss()
        
        self.num_vehicle = n_Veh
        self.action_all_with_power = np.zeros([self.num_vehicle, 3, 2],dtype = 'float')   # this is actions that taken by V2V links with power using test
        self.action_all_with_power_training = np.zeros([self.num_vehicle, 3, 2],dtype = 'float')   # this is actions that taken by V2V links with power for using train
        
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(DEVICE)
        
        
    def get_action(self, state):
        with torch.no_grad():
            a = self.actor(state)
        return a

    def update(self, state, action, reward, next_state, done):
        s, a, r, ns = state, action, reward, next_state

        # compute critic loss and update the critic parameters
        with torch.no_grad():
            #주목할점 actor target network로 action을 진행함
            critic_target = r + self.gamma * self.critic_target(ns, self.actor_target(ns)) * (1 - done)
        critic_loss = self.criteria(self.critic(s, a), critic_target)

        #critic 업데이트
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        #actor 업데이트
        # compute actor loss and update the actor parameters
        actor_loss = -self.critic(s, self.actor(s)).mean()  # !!!! Impressively simple
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
