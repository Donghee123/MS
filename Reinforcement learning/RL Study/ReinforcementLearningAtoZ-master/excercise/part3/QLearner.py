import torch
import torch.nn as nn
import numpy as np

#register_buffer 기능을 쓰려고 nn.Module 상속 받음
class NaiveDQN(nn.Module):

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 qnet: nn.Module,
                 lr: float,
                 gamma: float,
                 epsilon: float):
        super(NaiveDQN, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.qnet = qnet #파라미터 weight를 저장하고있는 네트워
        self.lr = lr
        self.gamma = gamma
        self.opt = torch.optim.Adam(params=self.qnet.parameters(), lr=lr)
        self.register_buffer('epsilon', torch.ones(1) * epsilon) #현재상태의 입실론을 저장하여 로드해서 사용할 수 있게끔 하는 방법

        self.criteria = nn.MSELoss() # (TD Target - 현재상태의 Q값)의 MSE(Mean Square Error)를 구해줌

    def get_action(self, state):
        #qnet은 MLP임 input dims : 4, output dims : 2
        qs = self.qnet(state)  # Notice that qs is 2d tensor [batch x action]
        """
        qs는 Q 네트워크의 아웃풋 값임 주의 할것은 qs는 2차원의 텐서인데
        1. Batch 사이즈를 의미
        2. action들의 값을 의미
        """
        #현재 모델이 train 모드라면 입실론 그리디 정책 이용
        if self.train:  # epsilon-greedy policy
            prob = np.random.uniform(0.0, 1.0, 1)
            
            #
            if torch.from_numpy(prob).float() <= self.epsilon:  # random 수행
                action = np.random.choice(range(self.action_dim))
            else:  # greedy 수행
                action = qs.argmax(dim=-1)
        #현재 모델이 eval 모드라면 그리디 정책 이용
        else:  # greedy policy  
            action = qs.argmax(dim=-1)
        return int(action)

    def update_sample(self, state, action, reward, next_state, done):
        s, a, r, ns = state, action, reward, next_state
        # Q-Learning target
        q_max, _ = self.qnet(next_state).max(dim=-1) #Q 네트워크를 이용해 다음 상태에 대한 다음 액션을 구해줌 
        
        #Q 네트워크로 뽑은 다음 상태 Q max값과 적용한 리워드값과 감마값으로 q_target 계산
        q_target = r + self.gamma * q_max * (1 - done)

        # detach는 Gradient를 트래킹하지 않기 위해서 함. DQN할때 무조건 필요!
        # Don't forget to detach `td_target` from the computational graph
        q_target = q_target.detach()
        q_target = q_target.to('cuda')
        q_selected = self.qnet(s)[0, a].to('cuda')
        # Or you can follow a better practice as follows:
        """
        detach와 비슷한 기능을 함.  Gradient를 트래킹하지 않는 방법
        with torch.no_grad():
            q_max, _ = self.qnet(next_state).max(dim=-1)
            q_target = r + self.gamma * q_max * (1 - done)
        """

        """        
        self.qnet(s)[0, a] 실제 행동에 대한 q값을 가져옴 (a는 정수 즉 선택한 결과일 뿐임 실제 q값은 self.qnet(s)[0, a]에 저장되어 있)
        q_target 계산한 q_target값임 
        loss함수의 파라미터는 항상 앞에가 prediction 데이터 두번째는 target 데이터임
        """
        loss = self.criteria(q_selected, q_target) 
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()


if __name__ == '__main__':

    import gym
    from src.part3.MLP import MultiLayerPerceptron as MLP


    class EMAMeter:

        def __init__(self,
                     alpha: float = 0.5):
            self.s = None
            self.alpha = alpha

        def update(self, y):
            if self.s is None:
                self.s = y
            else:
                self.s = self.alpha * y + (1 - self.alpha) * self.s


    env = gym.make('CartPole-v1')
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.n

    qnet = MLP(input_dim=s_dim,
               output_dim=a_dim,
               num_neurons=[128],
               hidden_act='ReLU',
               out_act='Identity')

    agent = NaiveDQN(state_dim=s_dim,
                     action_dim=a_dim,
                     qnet=qnet,
                     lr=1e-4,
                     gamma=1.0,
                     epsilon=1.0)

    n_eps = 10000
    print_every = 500
    ema_factor = 0.5
    ema = EMAMeter(ema_factor)

    for ep in range(n_eps):
        env.reset()  # restart environment
        cum_r = 0
        while True:
            s = env.state
            s = torch.tensor(s).float().view(1, 4)  # convert to torch.tensor
            a = agent.get_action(s)
            ns, r, done, info = env.step(a)

            ns = torch.tensor(ns).float()  # convert to torch.tensor
            agent.update_sample(s, a, r, ns, done)
            cum_r += r
            if done:
                ema.update(cum_r)

                if ep % print_every == 0:
                    print("Episode {} || EMA: {} || EPS : {}".format(ep, ema.s, agent.epsilon))

                if ep >= 150:
                    agent.epsilon *= 0.999
                break
    env.close()
