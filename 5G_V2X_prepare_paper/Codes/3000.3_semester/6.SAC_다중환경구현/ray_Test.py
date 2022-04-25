import ray
import datetime
import time
import pandas as pd
from Environment import *
from gym import spaces
import numpy as np
import torch
from sac import SAC
import argparse
from replay_memory import ReplayMemory

print(ray.__version__)

# Ray Task    
@ray.remote
def print_current_datetime():
    time.sleep(0.3)
    current_datetime = datetime.datetime.now()
    return current_datetime

ray.init()

@ray.remote
def random_game(env, vehicle, agent, memory):
    print(f'{env} init!')
    env.new_random_game(vehicle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')

    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--eval', type=bool, default=True,
                        help='Evaluates a policy a policy every 10 episode (default: True)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='G', #0.0003
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                        help='random seed (default: 123456)')
    parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                        help='maximum number of steps (default: 1000000)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 500)')


    parser.add_argument('--target_update_interval', type=int, default=10, metavar='N', # 1
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')


    #테스트 관련 하이퍼파라미터==============================================================

    parser.add_argument('--cuda', action="store_true",default=True,
                        help='run on CUDA (default: False)')

    parser.add_argument('--batch_size', type=int, default=256, metavar='N', # 256
                        help='batch size (default: 256)')

    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N', # 1
                        help='model updates per simulator step (default: 1)')

    parser.add_argument('--envs_per_process', type=int, default=4, metavar='N', # 1
                        help='set environment per process (default: 4)')

    #처음에 랜덤 선택하는 횟수를 지정함
    parser.add_argument('--start_steps', type=int, default=10000, metavar='N',  # 10000
                        help='Steps sampling random actions (default: 10000)')

    parser.add_argument('--train_step', type=int, default=40000, metavar='N',  # 40000
                        help='Set train step (default: 40000)')

    parser.add_argument('--test_step', type=int, default=2000, metavar='N',  # 2000
                        help='Set test interval step (default: 2000)')
    #======================================================================================

    args = parser.parse_args()


    up_lanes = [3.5/2, 3.5/2 + 3.5, 250+3.5/2,
                    250+3.5+3.5/2, 500+3.5/2, 500+3.5+3.5/2]
    down_lanes = [250-3.5-3.5/2, 250-3.5/2, 500-3.5 -
                    3.5/2, 500-3.5/2, 750-3.5-3.5/2, 750-3.5/2]
    left_lanes = [3.5/2, 3.5/2 + 3.5, 433+3.5/2,
                    433+3.5+3.5/2, 866+3.5/2, 866+3.5+3.5/2]
    right_lanes = [433-3.5-3.5/2, 433-3.5/2, 866-3.5 -
                    3.5/2, 866-3.5/2, 1299-3.5-3.5/2, 1299-3.5/2]

    width = 750
    height = 1299
    nVeh = 20


    # Agent
    statespaceSize = 82
    action_min_list = [0.0 for _ in range(20)]
    action_min_list.append(-10.0)

    action_max_list = [1.0 for _ in range(20)]
    action_max_list.append(23.0)

    action_space = spaces.Box(
            np.array(action_min_list), np.array(action_max_list), dtype=np.float32)


    


    envs = [Environ(down_lanes, up_lanes, left_lanes,
                    right_lanes, width, height, nVeh) for _ in range(3)] # V2X 환경 생성

    agent = SAC(statespaceSize, action_space, args, envs[0])
    memory = ReplayMemory(args.replay_size, args.seed)

    for env in envs:
        random_game.remote(env,nVeh, agent, memory)