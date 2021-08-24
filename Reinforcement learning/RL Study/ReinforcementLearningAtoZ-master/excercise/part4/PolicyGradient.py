import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


class REINFORCE(nn.Module):

    def __init__(self,
                 policy: nn.Module,
                 gamma: float = 1.0,
                 lr: float = 0.0002):
        super(REINFORCE, self).__init__()
        self.policy = policy  # make sure that 'policy' returns logits!
        self.gamma = gamma
        self.opt = torch.optim.Adam(params=self.policy.parameters(),
                                    lr=lr)
        
        #나중에 update 함수에서 log 계산할때 0이하로 가는것을 방지함
        self._eps = 1e-25

    def get_action(self, state):
        
        #편미분을 계산하지 않는 연산 동작을 의미
        with torch.no_grad():
            #상태 인풋
            logits = self.policy(state)
            
            #Categorical 이라는 클래스의 인풋으로 logits을 사용 후 distribution 아웃
            """            
            Categorical은 policy의 아웃풋을 다항 분포로 만들어줌
            출력의 합이 1이 되도록 하는 것을 의미함 (Softmax 와 동일)           
            """
            dist = Categorical(logits=logits)
            
            #distribution에서 샘플을 통해 action을 구함
            a = dist.sample()  # sample action from softmax policy
        return a

    @staticmethod
    def _pre_process_inputs(episode):
        states, actions, rewards = episode

        # assume inputs as follows
        # s : torch.tensor [num.steps x state_dim]
        # a : torch.tensor [num.steps]
        # r : torch.tensor [num.steps]

        # reversing inputs
        states = states.flip(dims=[0])
        actions = actions.flip(dims=[0])
        rewards = rewards.flip(dims=[0])
        return states, actions, rewards

    def update(self, episode):
        # sample-by-sample update version of REINFORCE
        # sample-by-sample update version is highly inefficient in computation
        states, actions, rewards = self._pre_process_inputs(episode)

        g = 0
        for s, a, r in zip(states, actions, rewards):
            g = r + self.gamma * g
            
            #출력값 합이 1이 아닌 actions을 받음
            logits=self.policy(s)          
            
            #dist는 합이 1인 actions를 받음
            dist = Categorical(logits)
            
            #dist중 실제 수행했던 action a의 확률을 가져옴 즉 정책에서 특정 state s에서 action a의 확률을 직접 가져옴
            #가치기반 강화학습에서는 Action value function을 가져왔었
            prob = dist.probs[a]

            # Don't forget to put '-' in the front of pg_loss !!!!!!!!!!!!!!!!
            # the default behavior of pytorch's optimizer is to minimize the targets
            # add 'self_eps' to prevent numerical problems of logarithms
            # loss 계산 : 
            #           prob가 0이 되는것은 방자하여 self._eps 더함
            #           -을 적용한 이유 minimizer에서 maximizer 적으로 바꾸기 위해
            pg_loss = - torch.log(prob + self._eps) * g

            self.opt.zero_grad()

            pg_loss.backward()
            self.opt.step()

    def update_episode(self, episode, use_norm=False):
        # batch update version of REINFORCE
        states, actions, rewards = self._pre_process_inputs(episode)

        # compute returns
        returns = []
        g = 0
        for r in rewards:
            g = r + self.gamma * g
            returns.append(g)
        returns = torch.tensor(returns)

        if use_norm:
            returns = (returns - returns.mean()) / (returns.std() + self._eps)

        # batch computation of action probabilities
        dist = Categorical(logits=self.policy(states))
        prob = dist.probs[range(states.shape[0]), actions]

        self.opt.zero_grad()

        # compute policy gradient loss
        pg_loss = - torch.log(prob + self._eps) * returns  # [num. steps x 1]
        pg_loss = pg_loss.mean()  # [1]
        pg_loss.backward()

        self.opt.step()

    def update_episodes(self, states, actions, returns, use_norm=False):
        # episode batch update version of REINFORCE

        if use_norm:
            returns = (returns - returns.mean()) / (returns.std() + self._eps)

        dist = Categorical(logits=self.policy(states))
        prob = dist.probs[range(states.shape[0]), actions]

        self.opt.zero_grad()

        # compute policy gradient loss
        pg_loss = - torch.log(prob + self._eps) * returns.squeeze()  # [num. steps x 1]
        pg_loss = pg_loss.mean()  # [1]
        pg_loss.backward()

        self.opt.step()


if __name__ == '__main__':
    import gym
    import torch

    from src.part3.MLP import MultiLayerPerceptron as MLP
    from src.part4.PolicyGradient import REINFORCE
    from src.common.train_utils import EMAMeter, to_tensor
    from src.common.memory.episodic_memory import EpisodicMemory

    env = gym.make('CartPole-v1')
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.n

    net = MLP(s_dim, a_dim, [128])
    agent = REINFORCE(net)
    ema = EMAMeter()
    memory = EpisodicMemory(max_size=100, gamma=1.0)

    n_eps = 10000
    update_every = 1
    print_every = 50

    for ep in range(n_eps):
        s = env.reset()
        cum_r = 0

        states = []
        actions = []
        rewards = []

        while True:
            s = to_tensor(s, size=(1, 4))
            a = agent.get_action(s)
            ns, r, done, info = env.step(a.item())

            # preprocess data
            r = torch.ones(1, 1) * r
            done = torch.ones(1, 1) * done

            memory.push(s, a, r, torch.tensor(ns), done)

            s = ns
            cum_r += r
            if done:
                break

        ema.update(cum_r)
        if ep % print_every == 0:
            print("Episode {} || EMA: {} ".format(ep, ema.s))

        if ep % update_every == 0:
            s, a, _, _, done, g = memory.get_samples()
            agent.update_episodes(s, a, g, use_norm=False)
            memory.reset()
