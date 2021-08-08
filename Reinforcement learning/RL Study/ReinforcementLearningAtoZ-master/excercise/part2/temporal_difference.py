import numpy as np


class TDAgent:

    def __init__(self,
                 gamma: float,
                 num_states: int,
                 num_actions: int,
                 epsilon: float,
                 lr: float,
                 n_step: int):
        self.gamma = gamma
        self.num_states = num_states
        self.num_actions = num_actions
        self.lr = lr
        self.epsilon = epsilon
        self.n_step = n_step

        # Initialize state value function V and action value function Q
        self.v = None
        self.q = None
        self.reset_values()

        # Initialize "policy Q"
        # "policy Q" is the one used for policy generation.
        self._policy_q = None
        self.reset_policy()

    def reset_values(self):
        self.v = np.zeros(shape=self.num_states)
        self.q = np.zeros(shape=(self.num_states, self.num_actions))

    def reset_policy(self):
        self._policy_q = np.zeros(shape=(self.num_states, self.num_actions))

    def get_action(self, state):
        prob = np.random.uniform(0.0, 1.0, 1)
        # e-greedy policy over Q
        if prob <= self.epsilon:  # random
            action = np.random.choice(range(self.num_actions))
        else:  # greedy
            action = self._policy_q[state, :].argmax()
        return action

    def update(self, episode):
        states, actions, rewards = episode
        ep_len = len(states)

        states += [0] * (self.n_step + 1)  # append dummy states
        rewards += [0] * (self.n_step + 1)  # append dummy rewards
        dones = [0] * ep_len + [1] * (self.n_step + 1)

        """
        kernel 역할
        사이즈가 5인 np.array
        g값을 더할때 discount factor 역할을함
        ex)
        kernel[0] = 0.9 ^ 1
        kernel[1] = 0.9 ^ 2
        kernel[2] = 0.9 ^ 3
        kernel[3] = 0.9 ^ 4
        kernel[4] = 0.9 ^ 5       
        """
        kernel = np.array([self.gamma ** i for i in range(self.n_step)])
        
        for i in range(ep_len):
            s = states[i]
            ns = states[i + self.n_step]
            done = dones[i]
            
            #reward 배열과 kernel의 값을 한번에 곱하고 모두 더함 -> 코드가 쉬워짐
            #G9 = R10 + rR11 + r^2V(S11)에서 
            #R10 + rR11 부분
            g = np.sum(rewards[i: i + self.n_step] * kernel)    
                
            #r^2V(S11) 을 추가로 더한 부분
            #Terminal state면 done값을 1로 지정 후 0으로 처리함
            g += (self.gamma ** self.n_step) * self.v[ns] * (1-done)
            
            #state value function 업데이트
            self.v[s] += self.lr * (g - self.v[s])

    def sample_update(self, state, action, reward, next_state, done):
        # done은 터미널일때만 1로 나오고 이때 (1 - done)은 0이 됨 -> 다음 상태의 state value function이 0이됨
        # 터미널 state의 state value function은 0임을 의미.
        td_target = reward + self.gamma * self.v[next_state] * (1 - done)
        self.v[state] += self.lr * (td_target - self.v[state])

    def decaying_epsilon(self, factor):
        self.epsilon *= factor


class SARSA(TDAgent):

    def __init__(self,
                 gamma: float,
                 num_states: int,
                 num_actions: int,
                 epsilon: float,
                 lr: float):
        super(SARSA, self).__init__(gamma=gamma,
                                    num_states=num_states,
                                    num_actions=num_actions,
                                    epsilon=epsilon,
                                    lr=lr,
                                    n_step=1)

    def get_action(self, state):
        prob = np.random.uniform(0.0, 1.0, 1)
        # e-greedy policy over Q
        if prob <= self.epsilon:  # random
            action = np.random.choice(range(self.num_actions))
        else:  # greedy
            action = self.q[state, :].argmax()
        return action

    def update_sample(self, state, action, reward, next_state, next_action, done):
        s, a, r, ns, na = state, action, reward, next_state, next_action

        # SARSA target
        # Refer to the lecture note <Part02 Chapter04 L04 TD control:SARSA> page 4
        td_target = r + self.gamma * self.q[ns,na] * (1 - done) #tdtarget : 다음 상태의 행동까지 구한 것
        self.q[s, a] += self.lr * (td_target - self.q[s, a]) #  현재 상태, 액션의 q값 += 학습률*(tdtarget - 현재 상태, 액션의 q값)


class QLearner(TDAgent):

    def __init__(self,
                 gamma: float,
                 num_states: int,
                 num_actions: int,
                 epsilon: float,
                 lr: float):
        super(QLearner, self).__init__(gamma=gamma,
                                       num_states=num_states,
                                       num_actions=num_actions,
                                       epsilon=epsilon,
                                       lr=lr,
                                       n_step=1)

    def get_action(self, state, mode='train'):
        if mode == 'train':
            prob = np.random.uniform(0.0, 1.0, 1)
            # e-greedy policy over Q
            if prob <= self.epsilon:  # random
                action = np.random.choice(range(self.num_actions))
            else:  # greedy
                action = self.q[state, :].argmax()
        else:
            action = self.q[state, :].argmax()
        return action

    def update_sample(self, state, action, reward, next_state, done):
        s, a, r, ns = state, action, reward, next_state
        # Q-Learning target
        # Refer to the lecture note <Part02 Chapter05 L02 Off-policy TD control and Q-Learning> page 7
        td_target = "Fill this line"
        self.q[s, a] += self.lr * (td_target - self.q[s, a])


def run_episode(env, agent):
    env.reset()
    while True:
        state = env.observe()
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        agent.update_sample(state, action, reward, next_state, done)

        if done:
            break
