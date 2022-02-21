"""
메모리를 계속 쌓았다가
done이 나오면 return Gt를 계산함.
결과는 self.returns 에서 쓰임.
"""
class Trajectory:
    def __init__(self, gamma: float):
        self.gamma = gamma
        self.states = list()
        self.actions = list()
        self.rewards = list()
        self.next_states = list()
        self.dones = list()

        self.length = 0
        self.returns = None
        self._discounted = False

    def push(self, state, action, reward, next_state, done):
        
        #done이 2번 이상 들어오면 못하게함.
        if done and self._discounted:
            raise RuntimeError("done is given at least two times!")

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.length += 1

        #done이 True이면 reward들을 이용해 Gt를 계산함.
        if done and not self._discounted:
            # compute returns
            self.compute_return()
            
    #지금까지 모은 reward를 계산해서 반환함.
    def compute_return(self):
        rewards = self.rewards
        returns = list()

        g = 0
        # iterating returns in inverse order
        # 역순으로 계산함.
        for r in rewards[::-1]:
            g = r + self.gamma * g
            returns.insert(0, g)
            
        self.returns = returns
        self._discounted = True

    def get_samples(self):
        return self.states, self.actions, self.rewards, self.next_states, self.dones, self.returns
