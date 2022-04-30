import argparse
from gym import spaces
import numpy as np
import torch
from sac import SAC
from Environment import *
from replay_memory import ReplayMemory
import os
import wandb
import ray
import copy

def play(self, actor_path, critic_path , n_step = 100, n_episode = 20, test_ep = None):
        
        self.load_model(actor_path = actor_path, critic_path = critic_path)
        number_of_game = n_episode
        V2I_Rate_list = np.zeros(number_of_game)
        V2V_Rate_list = np.zeros(number_of_game)
        Fail_percent_list = np.zeros(number_of_game)  
        self.training = False


        for game_idx in range(number_of_game):
            self.env.new_random_game(self.num_vehicle)
            test_sample = n_step
            Rate_list = []
            Rate_list_V2V = []
            
            print('test game idx:', game_idx)
            print('The number of vehicle is ', len(self.env.vehicles))
            
            time_left_list = []


            for k in range(test_sample):
                action_temp = self.action_all_with_power.copy()
                for i in range(len(self.env.vehicles)):
                    self.action_all_with_power[i, :, 0] = -1
                    sorted_idx = np.argsort(self.env.individual_time_limit[i, :])
                    for j in sorted_idx:
                        state_old = self.get_state([i, j], isTraining = False)
                        action = self.select_action(state_old, evaluate=True)

                        selected_resourceBlock , fselected_powerdB = self.ConvertToRealAction(action)
                        
                        self.action_all_with_power[i, j, 0] = selected_resourceBlock
                        self.action_all_with_power[i, j, 1] = fselected_powerdB
                                
                    #시뮬레이션 차량의 갯수 / 10 만큼 action이 정해지면 act를 수행함.
                    if i % (len(self.env.vehicles) / 10) == 1:
                        action_temp = self.action_all_with_power.copy()
                        returnV2IReward, returnV2VReward, fail_percent = self.env.act_asyn(action_temp)
                        Rate_list.append(np.sum(returnV2IReward))
                        Rate_list_V2V.append(np.sum(returnV2VReward))
                        
            V2I_Rate_list[game_idx] = np.mean(np.asarray(Rate_list))
            V2V_Rate_list[game_idx] = np.mean(np.asarray(Rate_list_V2V))
            
            Fail_percent_list[game_idx] = fail_percent

            print('Mean of the V2I rate is that ', np.mean(V2I_Rate_list[0:game_idx] ))
            print('Mean of the V2V rate is that ', np.mean(V2V_Rate_list[0:game_idx] ))
            print('Mean of Fail percent is that ',fail_percent, np.mean(Fail_percent_list[0:game_idx]))
            
        print('The number of vehicle is ', len(self.env.vehicles))
        print('Mean of the V2I rate is that ', np.mean(V2I_Rate_list))
        print('Mean of the V2V rate is that ', np.mean(V2V_Rate_list))
        print('Mean of Fail percent is that ', np.mean(Fail_percent_list))
        
        return np.mean(V2I_Rate_list), np.mean(V2V_Rate_list),np.mean(Fail_percent_list)

@ray.remote
def random_game(env, nvehicle):
    env.new_random_game(nvehicle)
    return env

@ray.remote
def renew_neighbor(env):
    env.renew_neighbor()
    return env

@ray.remote
def get_state(env,idx, isTraining):
    return env.get_state([idx[0],idx[1]], isTraining = isTraining) 

@ray.remote
def act_for_training(env, idx):
    return env.act_for_training(env.action_all_with_power_training, idx)


def parrel_get_state(envs,idx, isTraining):
    datas = [get_state.remote(env, idx, isTraining) for env in envs]
    states = ray.get(datas)    
    return states

def parrel_random_games(envs, nvehicle):
    envs = [random_game.remote(env, nvehicle) for env in envs]
    envs = ray.get(envs)
    return copy.deepcopy(envs)
        
def parrel_renew_neighbors(envs):
    envs = [renew_neighbor.remote(env) for env in envs]
    envs = ray.get(envs)
    return copy.deepcopy(envs)

def parrel_act_for_training(envs, idx):
    ray.put(envs)
    datas = [act_for_training.remote(env, idx) for env in envs]
    rewards_listOfchanged = ray.get(datas)  
    
    envs = copy.deepcopy(envs)

    listOfReturnEnvs = []
    listOfRewards = []
    for nIndex, env in enumerate(envs): 
        demand = rewards_listOfchanged[nIndex][1][0]
        test_time_count = rewards_listOfchanged[nIndex][1][1]
        individual_time_limit = rewards_listOfchanged[nIndex][1][2]
        env.demand = demand
        env.test_time_count = test_time_count
        env.individual_time_limit = individual_time_limit
        env.renew_positions()
        env.renew_channels_fastfading()
        env.Compute_Interference(env.action_all_with_power_training)
        listOfReturnEnvs.append(env)
        listOfRewards.append(rewards_listOfchanged[nIndex][0])

    #modify env
      
    return listOfReturnEnvs, listOfRewards

def train(agent,memory,args, envs, testenv): 
        
    #공유 agent
    agent = agent

    #공유 memory
    memory = memory
    
    #공유 하이퍼파라미터
    args = args

    # 개별 V2X 환경 적용
    env = envs[0]  # V2X 환경 1 ref

    updates = 0
    update_count= 0
    total_loss, total_q = 0.,0.

    total_numsteps = 0
    
    envs = parrel_random_games(envs, 20)
      
    rewardloggingData = []
    
    for step in (range(0, args.train_step)): # need more configuration #40000
            print(step)

            if step == 0:                   # initialize set some varibles
                update_count = 0
                total_loss, total_q = 0., 0.
                
            # prediction
            if (step % 2000 == 1):
                envs = parrel_renew_neighbors(envs)
                
            
            training = True
            fReward_sum = 0
                        
            #현재 상태에서 모든 차량이 선택을 수행함.
            for k in range(1):
                #i번째 송신 차량
                for i in range(len(envs[0].vehicles)):
                    #i번째 송신 차량에서 전송한 신호를 수신하는 j번째 수신 차량 
                    for j in range(3): 
                        
                        # i번째 차량에서 j번째 차량으로 데이터를 전송할때 state를 가져옴.
                        state_olds = parrel_get_state(envs,[i,j],isTraining = True)
                        
                        #state를 보고 action을 정함
                        #action은 선택한 power level, 선택한 resource block 정보를 가짐
                        # 랜덤 선택
                        listsof_action = []
                        if args.start_steps > total_numsteps:
                            #for parrel
                            
                            for _ in range(len(envs)):
                                listofRandom_actions = [random.random() for _ in range(20)]
                                listofRandom_actions.append(random.uniform(-10.0, 23.0))
                                action = np.array(listofRandom_actions)
                                listsof_action.append(action)
                        else:
                            
                            for state_old in state_olds:
                                listsof_action.append(agent.select_action(state_old))

                        # 업데이트 
                        if len(memory) > args.batch_size:
                            # Number of updates per step in environment
                            for z in range(args.updates_per_step):
                                # Update parameters of all the networks
                                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)
                                updates += 1                            

                        total_numsteps+=1
                        
                        #선택 selected_resourceBlock, selected_powerdB     
                        listOfselected_resourceBlock = []
                        listOfselected_fselected_powerdB = []
                        for action in listsof_action:            
                            selected_resourceBlock , fselected_powerdB = agent.ConvertToRealAction(action)
                            listOfselected_resourceBlock.append(selected_resourceBlock)
                            listOfselected_fselected_powerdB.append(fselected_powerdB)

                        for index, env in enumerate(envs):
                            env.merge_action([i,j],int(listOfselected_resourceBlock[index]), listOfselected_fselected_powerdB[index], bIsTrainMode = True)

                                         
                        #선택한 power level과 resource block을 기반으로 reward를 계산함.
                        envs, listOfReward_train = parrel_act_for_training(envs, [i,j]) 

                        fReward_sum += np.mean(listOfReward_train)
                        
                        state_news = parrel_get_state(envs,[i,j],isTraining = True)

                        for nIndex in range(len(state_news)):            
                            memory.push(state_olds[nIndex].reshape(82), listsof_action[nIndex], np.array([listOfReward_train[nIndex]]), state_news[nIndex].reshape(82),np.array([1])) # Append transition to memory

            wandb.log({f"SAC_one_step_reward": fReward_sum})
                                                                   
            if (step % args.test_step == 0) and (step > 0):
                # testing 
                training = False

                number_of_game = 10

                if (step % 10000 == 0) and (step > 0):
                    number_of_game = 50 

                if (step == 38000):
                    number_of_game = 100     

                V2I_Rate_list = np.zeros(number_of_game)
                V2V_Rate_list = np.zeros(number_of_game)
                Fail_percent_list = np.zeros(number_of_game)
                for game_idx in range(number_of_game):
                    
                    testenv.new_random_game(env.n_Veh)
                    test_sample = 200
                    V2IRate_list = []
                    V2VRate_list = []
                    print('test game idx:', game_idx)
                    for k in range(test_sample):
                        action_temp = testenv.action_all_with_power.copy()
                        for i in range(len(env.vehicles)):
                            testenv.action_all_with_power[i,:,0] = -1
                            sorted_idx = np.argsort(testenv.individual_time_limit[i,:])          
                            for j in sorted_idx:                   
                                state_old = testenv.get_state([i,j], isTraining = False)                               
                                action = agent.select_action(state_old, evaluate=True)

                                selected_resourceBlock , fselected_powerdB = agent.ConvertToRealAction(action)

                                testenv.action_all_with_power[i, j, 0] = selected_resourceBlock
                                testenv.action_all_with_power[i, j, 1] = fselected_powerdB
                                
                            if i % (len(env.vehicles)/10) == 1:
                                action_temp = testenv.action_all_with_power.copy()
                                V2IRate, V2VRate, percent = testenv.act_asyn(action_temp) #self.action_all)            
                                V2IRate_list.append(np.sum(V2IRate))
                                V2VRate_list.append(np.sum(V2VRate))
                        #print("actions", self.action_all_with_power)
                    V2I_Rate_list[game_idx] = np.mean(np.asarray(V2IRate_list))
                    V2V_Rate_list[game_idx] = np.mean(np.asarray(V2VRate_list))
                    Fail_percent_list[game_idx] = percent
                    #print("action is", self.action_all_with_power)
                    print(f'failure probability is, ', percent)
                    #print('action is that', action_temp[0,:])
                    

                wandb.log({f"SAC_V2I_Rete": np.mean(V2I_Rate_list), f"SAC_V2V_Rete": np.mean(V2V_Rate_list), f"SAC_V2V_fail_pecent": np.mean(Fail_percent_list)})

                agent.save_model('V2X_Model_' + str(step) + '_' + str(np.mean(V2I_Rate_list) + np.mean(V2V_Rate_list)) + '_' + str(np.mean(Fail_percent_list)))
                print (f'The number of vehicle is ', len(env.vehicles))
                print (f'Mean of the V2I rate + V2V rate is that ', np.mean(V2I_Rate_list) + np.mean(V2V_Rate_list))
                print (f'Mean of the V2I rate is that ', np.mean(V2I_Rate_list))
                print (f'Mean of the V2V rate is that ', np.mean(V2V_Rate_list))
                print(f'Mean of Fail percent is that ', np.mean(Fail_percent_list))    

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
    parser.add_argument('--start_steps', type=int, default=100000, metavar='N',  # 10000
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
    # Start 초기화 부분
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    #ray instance init
    ray.init()

    # Agent
    statespaceSize = 82
    action_min_list = [0.0 for _ in range(20)]
    action_min_list.append(-10.0)

    action_max_list = [1.0 for _ in range(20)]
    action_max_list.append(23.0)

    action_space = spaces.Box(
        np.array(action_min_list), np.array(action_max_list), dtype=np.float32)

    testenv = Environ(down_lanes, up_lanes, left_lanes,
                    right_lanes, width, height, nVeh)

    envs = [Environ(down_lanes, up_lanes, left_lanes,
                    right_lanes, width, height, nVeh) for _ in range(10)] # V2X 환경 생성

    wandb.init(config=args, project="Multi Environment V2V Resource Allocation by SAC")
    wandb.config["My pytorch SAC"] = "Multi Environment SAC Version 0.1"

    agent = SAC(statespaceSize, action_space, args, envs[0])
    memory = ReplayMemory(args.replay_size, args.seed)

    train(agent, memory, args, envs, testenv)



