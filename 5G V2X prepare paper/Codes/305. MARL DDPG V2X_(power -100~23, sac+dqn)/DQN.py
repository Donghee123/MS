import torch
import numpy as np
import torch.nn as nn


class DQN(nn.Module):

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 qnet: nn.Module,
                 qnet_target: nn.Module,
                 lr: float,
                 gamma: float,
                 epsilon: float):
        """
        :param state_dim: input state dimension
        :param action_dim: action dimension
        :param qnet: main q network
        :param qnet_target: target q network
        :param lr: learning rate
        :param gamma: discount factor of MDP
        :param epsilon: E-greedy factor
        """

        super(DQN, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.qnet = qnet
        self.lr = lr
        self.gamma = gamma
        self.opt = torch.optim.Adam(params=self.qnet.parameters(), lr=lr)
        
        #트레이닝 과정중에 입실론 저장을 위해 등록함
        self.register_buffer('epsilon', torch.ones(1) * epsilon)

        # target network related
        """
        학습한 qnet의 파라미터를
        qnet_target의 파라미터로 저장시킴
        """
        qnet_target.load_state_dict(qnet.state_dict())
        self.qnet_target = qnet_target
        
        
        """
         Mean-squared Error (MSE) Loss 단점 중 하나는 데이터의 outlier에 매우 취약하다는 것임.
         모종의 이유로 타겟하는 레이블 y(이 예제의 경우 td-target/q-learning target)에 noise 데이터가 섞여 있을 경우, 잘못된 y값을 추산하게 됨
         그 결과 네트워크의 파라미터가 sensitive하게 움직이게 됨.
         
         이런 현상은 q-learning의 학습 초기에 매우 빈번히 나타날 것으로 예상 가능함.
         
         이러한 문제를 조금이라도 완화하기 위해서 outlier에 덜 민감한 loss 함수를 사용했음.
         nn.SmmothL1Loss는
         
         if abs(x-y) < 1
            return 0.5 * abs(x - y)
         else if (abs(x-y) >= 1)
            return abs(x - y) - 0.5
        """
        
        self.criteria = nn.SmoothL1Loss()

    def get_action(self, state):
        #입실론 그리디 기반
        qs = self.qnet(state)
        prob = np.random.uniform(0.0, 1.0, 1)
        if torch.from_numpy(prob).float() <= self.epsilon:  # random
            action = np.random.choice(range(self.action_dim))
        else:  # greedy
            action = qs.argmax(dim=-1)
        return int(action)

    def update(self, state, action, reward, next_state, done):
        s, a, r, ns = state, action, reward, next_state

        # compute Q-Learning target with 'target network'
        with torch.no_grad():
            q_max, _ = self.qnet_target(ns).max(dim=-1, keepdims=True)
            q_target = r + self.gamma * q_max * (1 - done)

        q_val = self.qnet(s).gather(0, a)
        loss = self.criteria(q_val, q_target)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

def prepare_training_inputs(sampled_exps, device='cpu'):
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []
    for sampled_exp in sampled_exps:
        states.append(sampled_exp[0])
        actions.append(sampled_exp[1])
        rewards.append(sampled_exp[2])
        next_states.append(sampled_exp[3])
        dones.append(sampled_exp[4])

    states = torch.cat(states, dim=0).float().to(device)
    actions = torch.cat(actions, dim=0).to(device)
    rewards = torch.cat(rewards, dim=0).float().to(device)
    next_states = torch.cat(next_states, dim=0).float().to(device)
    dones = torch.cat(dones, dim=0).float().to(device)
    return states, actions, rewards, next_states, dones

