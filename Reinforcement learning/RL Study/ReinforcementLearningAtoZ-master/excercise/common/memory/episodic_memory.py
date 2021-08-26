import torch
from collections import deque
from common.memory.trajectory import Trajectory


class EpisodicMemory:

    def __init__(self, max_size: int, gamma: float):
        self.max_size = max_size  # maximum number of trajectories
        self.gamma = gamma
        self.trajectories = deque(maxlen=max_size)
        self._trajectory = Trajectory(gamma=gamma)

    #state, action, reward, nest_state, done을 __trajectory에 저장함.
    def push(self, state, action, reward, next_state, done):
        self._trajectory.push(state, action, reward, next_state, done)
        if done:
            self.trajectories.append(self._trajectory)
            self._trajectory = Trajectory(gamma=self.gamma)

    def reset(self):
        self.trajectories.clear()
        self._trajectory = Trajectory(gamma=self.gamma)
        
        
    def get_samples(self):
        states, actions, rewards, next_states, dones, returns = [], [], [], [], [], []
        #self.trajectories의 사이즈가 0보다 크다면
        while self.trajectories:
            traj = self.trajectories.pop()
            s, a, r, ns, done, g = traj.get_samples()
            states.append(torch.cat(s, dim=0))
            actions.append(torch.cat(a, dim=0))
            rewards.append(torch.cat(r, dim=0))
            next_states.append(torch.cat(ns, dim=0))
            dones.append(torch.cat(done, dim=0))
            returns.append(torch.cat(g, dim=0)) #trajectories에서 done이 True가 있어야 returns이 있음.

        states = torch.cat(states, dim=0)
        actions = torch.cat(actions, dim=0)
        rewards = torch.cat(rewards, dim=0)
        next_states = torch.cat(next_states, dim=0)
        dones = torch.cat(dones, dim=0)
        returns = torch.cat(returns, dim=0)

        return states, actions, rewards, next_states, dones, returns
