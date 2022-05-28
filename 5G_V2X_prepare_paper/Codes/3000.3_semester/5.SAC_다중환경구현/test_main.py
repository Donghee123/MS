import argparse
from gym import spaces
import numpy as np
import torch
from sac import SAC
from Environment import *
#from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory


#File 유틸 함수들    
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
     
def MakeCSVFile(strFolderPath, strFilePath, aryOfHedaers, aryOfDatas):
    strTotalPath = "%s\%s" % (strFolderPath,strFilePath)
    
    f = open(strTotalPath,'w', newline='')
    wr = csv.writer(f)
    wr.writerow(aryOfHedaers)
    
    for i in range(0,len(aryOfDatas)):
        wr.writerow(aryOfDatas[i])
    
    f.close()

def ConvertToRealAction(self, action):
        
        fselect_maxValue  = np.max(action[0:20])
        listofCandidateList_selRB = np.where(action[0:20] == fselect_maxValue)
        nindex_selectedRB = random.sample(list(listofCandidateList_selRB[0]),k=1)[0]

        fPower = action[20]
        return nindex_selectedRB, fPower

def play(agent, env, num_vehicle, n_step = 100, n_episode = 20):
        
        number_of_game = n_episode
        V2I_Rate_list = np.zeros(number_of_game)
        V2V_Rate_list = np.zeros(number_of_game)
        Fail_percent_list = np.zeros(number_of_game)  
        training = False

        for game_idx in range(number_of_game):
            env.new_random_game(num_vehicle)
            test_sample = n_step
            Rate_list = []
            Rate_list_V2V = []
            
            print('test game idx:', game_idx)
            print('The number of vehicle is ', len(env.vehicles))
            
            time_left_list = []


            for k in range(test_sample):
                action_temp = env.action_all_with_power.copy()
                for i in range(len(env.vehicles)):
                    env.action_all_with_power[i, :, 0] = -1
                    sorted_idx = np.argsort(env.individual_time_limit[i, :])
                    for j in sorted_idx:
                        state_old = env.get_state([i, j], isTraining = False)
                        action = agent.select_action(state_old, evaluate=True)

                        selected_resourceBlock , fselected_powerdB = ConvertToRealAction(action)
                        
                        env.action_all_with_power[i, j, 0] = selected_resourceBlock
                        env.action_all_with_power[i, j, 1] = fselected_powerdB
                                
                    #시뮬레이션 차량의 갯수 / 10 만큼 action이 정해지면 act를 수행함.
                    if i % (len(env.vehicles) / 10) == 1:
                        action_temp = env.action_all_with_power.copy()
                        returnV2IReward, returnV2VReward, fail_percent = env.act_asyn(action_temp)
                        Rate_list.append(np.sum(returnV2IReward))
                        Rate_list_V2V.append(np.sum(returnV2VReward))
                        
            V2I_Rate_list[game_idx] = np.mean(np.asarray(Rate_list))
            V2V_Rate_list[game_idx] = np.mean(np.asarray(Rate_list_V2V))
            
            Fail_percent_list[game_idx] = fail_percent

            print('Mean of the V2I rate is that ', np.mean(V2I_Rate_list[0:game_idx] ))
            print('Mean of the V2V rate is that ', np.mean(V2V_Rate_list[0:game_idx] ))
            print('Mean of Fail percent is that ',fail_percent, np.mean(Fail_percent_list[0:game_idx]))
            
        print('The number of vehicle is ', len(env.vehicles))
        print('Mean of the V2I rate is that ', np.mean(V2I_Rate_list))
        print('Mean of the V2V rate is that ', np.mean(V2V_Rate_list))
        print('Mean of Fail percent is that ', np.mean(Fail_percent_list))
        
        return np.mean(V2I_Rate_list), np.mean(V2V_Rate_list),np.mean(Fail_percent_list)

sumrateV2IList = []
sumrateV2VList = []

probabilityOfSatisfiedV2VList = []

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

# Environment
# env = NormalizedActions(gym.make(args.env_name))

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

# V2X 환경 적용
env = Environ(down_lanes, up_lanes, left_lanes,
              right_lanes, width, height, nVeh)  # V2X 환경 생성


# Start 초기화 부분
# env.seed(args.seed) 초기화 부분
# env.action_space.seed(args.seed) 초기화 부분

torch.manual_seed(args.seed)
np.random.seed(args.seed)
# End 초기화 부분


# Agent
statespaceSize = 82
action_min_list = [0.0 for _ in range(20)]
action_min_list.append(-10.0)

action_max_list = [1.0 for _ in range(20)]
action_max_list.append(23.0)

action_space = spaces.Box(
    np.array(action_min_list), np.array(action_max_list), dtype=np.float32)


envs = [Environ(down_lanes, up_lanes, left_lanes,
                    right_lanes, width, height, nVeh) for _ in range(1)] # V2X 환경 생성

agent = SAC(statespaceSize, action_space, args, envs[0])

# Memory
memory = ReplayMemory(args.replay_size, args.seed)

arrayOfVeh = [20,40,60,80,100]

actor_path = './models/testmodel/sac_actor_V2X_Model_'
critic_path = './models/testmodel/sac_critic_V2X_Model_'

for nVeh in arrayOfVeh:      
    Env = Environ(down_lanes,up_lanes,left_lanes,right_lanes, width, height,nVeh)
    Env.new_random_game(nVeh)
      
    agent = SAC(statespaceSize, action_space, args, Env)

    agent.load_model(actor_path = actor_path, critic_path= critic_path) 

    #학습 
    v2i_Sumrate, v2v_Sumrate, probability = agent.play(actor_path= actor_path, critic_path= critic_path ,n_step = 100, n_episode = 20, random_choice = False)
    
    sumrateV2IList.append(v2i_Sumrate)
    sumrateV2VList.append(v2v_Sumrate)
        
    probabilityOfSatisfiedV2VList.append(probability)
               

sumrateV2IListnpList = np.array(sumrateV2IList)
sumrateV2VListnpList = np.array(sumrateV2VList)
sumrateV2V_V2IListnpList = sumrateV2IListnpList + sumrateV2VListnpList
probabilityOfSatisfiedV2VnpList = np.array(probabilityOfSatisfiedV2VList)
  
print('V2I sumrate')
print(sumrateV2IListnpList)
print('V2V sumrate')
print(sumrateV2VListnpList)
print('V2V + V2I rate')
print(sumrateV2IListnpList + sumrateV2VListnpList)
print('Outage probability')
print(probabilityOfSatisfiedV2VnpList)

allData=[]
allData.append(sumrateV2IListnpList)
allData.append(sumrateV2VListnpList)
allData.append(sumrateV2V_V2IListnpList)
allData.append(probabilityOfSatisfiedV2VnpList)
allData = np.transpose(allData)
  
folderPath = './ResultData'
csvFileName = 'ResultData.csv'
createFolder(folderPath)
MakeCSVFile(folderPath, csvFileName, allData)
