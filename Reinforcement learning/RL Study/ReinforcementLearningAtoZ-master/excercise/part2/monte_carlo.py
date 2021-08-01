import numpy as np


class ExactMCAgent:
    """
    The exact Monte-Carlo agent.
    This agents performs value update as follows:
    V(s) <- s(s) / n(s)
    Q(s,a) <- s(s,a) / n(s,a)
    """

    def __init__(self,
                 gamma: float,
                 num_states: int,
                 num_actions: int,
                 epsilon: float):
        self.gamma = gamma
        self.num_states = num_states
        self.num_actions = num_actions
        self.epsilon = epsilon

        self._eps = 1e-10  # for stable computation of V and Q. NOT the one for e-greedy !

        # Initialize statistics
        self.n_v = None
        self.s_v = None
        self.n_q = None
        self.s_q = None
        self.reset_statistics()

        # Initialize state value function V and action value function Q
        self.v = None
        self.q = None
        self.reset_values()

        # Initialize "policy Q"
        # "policy Q" is the one used for policy generation.
        self._policy_q = None
        self.reset_policy()

    def reset_statistics(self):
        self.n_v = np.zeros(shape=self.num_states)
        self.s_v = np.zeros(shape=self.num_states)

        self.n_q = np.zeros(shape=(self.num_states, self.num_actions))
        self.s_q = np.zeros(shape=(self.num_states, self.num_actions))

    def reset_values(self):
        self.v = np.zeros(shape=self.num_states)
        self.q = np.zeros(shape=(self.num_states, self.num_actions))

    def reset_policy(self):
        self._policy_q = np.zeros(shape=(self.num_states, self.num_actions))

    def reset(self):
        self.reset_statistics()
        self.reset_values()
        self.reset_policy()

    def update(self, episode):
        '''
        바닐라 버전 MC
        episode 튜플의 데이터들 가져오기
        every visit을 할것임.
        MC의 rewards를 계산 할때 뒤에서 부터 계산하면 1번의 episode순환으로 가능
        코드에서는 reverser를 이용해서 배열의 순서를 뒤집음
        ex)
        s1 s2 s2 s4 s5 s1 s2 s4 sT
        
        s1를 기준으로 업데이터 한다할때
        s1 : r1 s2 : r2 s4 : r4 sT : rT
        s1 update값 1번 : r1 + r2 + r4 + rT
        s1 update값 2번 : r1 + r2 + r2 + r4 + r5 + s1 update값 1번(재사용)
        s1의 value function = (s1 update값 1번 + s1 update값 2번) / 2
        '''
        states, actions, rewards = episode

        # reversing the inputs!
        # for efficient computation of returns
        states = reversed(states)
        actions = reversed(actions)
        rewards = reversed(rewards)

        iter = zip(states, actions, rewards)
        cum_r = 0
        for s, a, r in iter:
            # Refer to the lecture note <Part02 Chapter03 L01 MC evaluation> page 11
            # compute the return of given state s
            # Notice that we are computing the returns in the reversed order!
            cum_r *= self.gamma
            cum_r += r
            
            self.n_v[s] += 1
            self.n_q[s, a] += 1

            self.s_v[s] += cum_r
            self.s_q[s, a] += cum_r

    def compute_values(self):
        self.v = self.s_v / (self.n_v + self._eps)
        self.q = self.s_q / (self.n_q + self._eps)

    def get_action(self, state):
        prob = np.random.uniform(0.0, 1.0, 1)
        # e-greedy policy over Q
        if prob <= self.epsilon:  # random
            action = np.random.choice(range(self.num_actions))
        else:  # greedy
            action = self._policy_q[state, :].argmax()
        return action

    def improve_policy(self):
        self._policy_q = self.q.copy()
        # self.reset_statistics()

    def decaying_epsilon(self, factor):
        self.epsilon *= factor


class MCAgent(ExactMCAgent):
    """
    The 'learning-rate' Monte-Carlo agent.
    This agents performs value update as follows:
    V(s) <- V(s) + lr * (Gt - V(s))
    Q(s,a) <- Q(s,a) + lr * (Gt - Q(s,a))
    """

    def __init__(self,
                 gamma: float,
                 num_states: int,
                 num_actions: int,
                 epsilon: float,
                 lr: float):
        super(MCAgent, self).__init__(gamma=gamma,
                                      num_states=num_states,
                                      num_actions=num_actions,
                                      epsilon=epsilon)
        self.lr = lr

     
    def update(self, episode):
        '''
        increamental 버전 MC
        episode 튜플의 데이터들 가져오기
        every visit을 할것임.
        MC의 rewards를 계산 할때 뒤에서 부터 계산하면 1번의 episode순환으로 가능
        코드에서는 reverser를 이용해서 배열의 순서를 뒤집음
        ex)
        s1 s2 s2 s4 s5 s1 s2 s4 sT
        
        s1를 기준으로 업데이터 한다할때
        s1 : r1 s2 : r2 s4 : r4 sT : rT
        s1 update값 1번 : r1 + r2 + r4 + rT
        s1 update값 2번 : r1 + r2 + r2 + r4 + r5 + s1 update값 1번(재사용)
        
        s1의 value function = s1의 value function + learning rate * (cumulative reward - s1의 value function)
        '''
        states, actions, rewards = episode

        # reversing the inputs!
        # for efficient computation of returns
        states = reversed(states)
        actions = reversed(actions)
        rewards = reversed(rewards)

        iter = zip(states, actions, rewards)
        cum_r = 0.0
        
        for s, a, r in iter:
            
            cum_r *= self.gamma
            cum_r += r

            # Implement learning rate version of MC policy evaluation
            # Refer to the lecture note <Part02 Chapter03 L01 MC evaluation> page 14           
            self.v[s] = self.v[s] + self.lr * (cum_r - self.v[s])
            self.q[s, a] = self.q[s, a] + self.lr * (cum_r - self.q[s,a])



