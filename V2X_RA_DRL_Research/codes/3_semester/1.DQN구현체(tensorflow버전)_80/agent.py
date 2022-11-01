from __future__ import print_function, division
import os
import time
import random
import numpy as np
from base import BaseModel
from replay_memory import ReplayMemory
from utils import save_pkl, load_pkl
import tensorflow as tf
import matplotlib.pyplot as plt
import wandb 
import argparse
import pickle
import _pickle as cPickle
import pdb

class Agent(BaseModel):
    def __init__(self, config, environment, sess):
        self.sess = sess
        self.weight_dir = 'weight'        
        self.env = environment
        #self.history = History(self.config)
        model_dir = './Model/a.model'
        self.memory = ReplayMemory(model_dir) 
        self.isCollectMemory = True
        self.max_step = 100000 
        
        self.replay_memory_size = self.memory.memory_size
        
        self.RB_number = 20
        self.num_vehicle = len(self.env.vehicles)
        self.action_all_with_power = np.zeros([self.num_vehicle, 3, 2],dtype = 'int32')   # this is actions that taken by V2V links with power
        self.action_all_with_power_training = np.zeros([20, 3, 2],dtype = 'int32')   # this is actions that taken by V2V links with power
        self.reward = []
        self.learning_rate = 0.01
        self.learning_rate_minimum = 0.0001
        self.learning_rate_decay = 0.96
        self.learning_rate_decay_step = 500000
        self.target_q_update_step = 100
        self.discount = 0.5
        self.double_q = True
        self.build_dqn()          
        self.V2V_number = 3 * len(self.env.vehicles)    # every vehicle need to communicate with 3 neighbors  
        self.training = True
        #self.actions_all = np.zeros([len(self.env.vehicles),3], dtype = 'int32')
    def merge_action(self, idx, action):
        self.action_all_with_power[idx[0], idx[1], 0] = action % self.RB_number
        self.action_all_with_power[idx[0], idx[1], 1] = int(np.floor(action/self.RB_number))

    def loadparameter(self, path):
        with open(path, 'rb') as f:
            obj = cPickle.load(f)
            print("  [*] load %s" % path)
            return obj
        
    def get_state(self, idx):
    # ===============
    #  Get State from the environment
    # =============
        vehicle_number = len(self.env.vehicles)
        
        #idx번째 차량이 전송하고자하는 v2v link의 resource block의 채널 상태를 보여줌
        V2V_channel = (self.env.V2V_channels_with_fastfading[idx[0],self.env.vehicles[idx[0]].destinations[idx[1]],:] - 80)/60
        
        #idx번째 차량이 전송하고자하는 v2i link의 resource block의 채널 상태를 보여줌
        V2I_channel = (self.env.V2I_channels_with_fastfading[idx[0], :] - 80)/60
        
        #이전 스탭에서 idx번째 차량이 전송하고자하는 v2v link의 resource block에서 살펴 볼 수 있는 Interference
        V2V_interference = (-self.env.V2V_Interference_all[idx[0],idx[1],:] - 60)/60
        #선택한 resource block
        NeiSelection = np.zeros(self.RB_number)
    
        #인접한 차량에게 전송할 power 선정
        for i in range(3):
            for j in range(3):
                if self.training:
                    NeiSelection[self.action_all_with_power_training[self.env.vehicles[idx[0]].neighbors[i], j, 0 ]] = 1
                else:
                    NeiSelection[self.action_all_with_power[self.env.vehicles[idx[0]].neighbors[i], j, 0 ]] = 1
                   
        for i in range(3):
            if i == idx[1]:
                continue
            if self.training:
                if self.action_all_with_power_training[idx[0],i,0] >= 0:
                    NeiSelection[self.action_all_with_power_training[idx[0],i,0]] = 1
            else:
                if self.action_all_with_power[idx[0],i,0] >= 0:
                    NeiSelection[self.action_all_with_power[idx[0],i,0]] = 1
                    
        time_remaining = np.asarray([self.env.demand[idx[0],idx[1]] / self.env.demand_amount])
        load_remaining = np.asarray([self.env.individual_time_limit[idx[0],idx[1]] / self.env.V2V_limit])
        #pdb.set_trace()
        #print('shapes', time_remaining.shape,load_remaining.shape)
        # V2I_channel : #idx번째 차량이 전송하고자하는 v2i link의 resource block의 채널 상태를 보여줌
        # V2V_interference : 이전 스탭에서 idx번째 차량이 전송하고자하는 v2v link의 resource block에서 볼 수 있는 Interference
        # V2V_channel : #idx번째 차량이 전송하고자하는 v2v link의 resource block의 채널 상태를 보여줌
        # 근접한 차량이 선택한 리소스 블록 상태
        # 남은 시간
        # 걸린 시간

        return np.concatenate((V2I_channel, V2V_interference, V2V_channel, NeiSelection, time_remaining, load_remaining))#,time_remaining))
        #return np.concatenate((V2I_channel, V2V_interference, V2V_channel, time_remaining, load_remaining))#,time_remaining))

    def predictwidthModel(self, model, s_t):
        # ==========================
        #  Select actions
        # ======================
        s_t = np.expand_dims(s_t, 0)
        action =  model.predict(s_t)
        actionIndex = np.argmax(action)
        return actionIndex

    def predict(self, s_t,  step, test_ep = False, random_choice = False):
        # ==========================
        #  Select actions
        # ======================
        if random_choice == True:
            action = np.random.randint(60)
            return action
        else:
            ep = 1/(step/1000000 + 1)
            if random.random() < ep and test_ep == False:   # epsion to balance the exporation and exploition
                action = np.random.randint(60)
            else:          
                action =  self.q_action.eval({self.s_t:[s_t]})[0] 
            return action
        
    def observe(self, prestate, state, reward, action):
        # -----------
        # Collect Data for Training 
        # ---------
        self.memory.add(prestate, state, reward, action) # add the state and the action and the reward to the memory
        #print(self.step)
        if self.step > 0:
            if self.step % 50 == 0:
                #print('Training')
                self.q_learning_mini_batch()            # training a mini batch
                #self.save_weight_to_pkl()
            if self.step % self.target_q_update_step == self.target_q_update_step - 1:
                #print("Update Target Q network:")
                self.update_target_q_network()           # ?? what is the meaning ??
    
    
    def train(self):  
        
        parser = argparse.ArgumentParser(description='PyTorch on DQN')

        parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
        parser.add_argument('--train_resume', default=1, type=int, help='train resume, using path : ./ddpg/resume model/actor, ./ddpg/resume model/critic')
        parser.add_argument('--hidden1', default=500, type=int, help='hidden1 num of first fully connect layer')
        parser.add_argument('--hidden2', default=250, type=int, help='hidden2 num of second fully connect layer')
        parser.add_argument('--hidden3', default=120, type=int, help='hidden3 num of first fully connect layer')
        parser.add_argument('--learning_rate', default=0.0001, type=float, help='learning rate')   
        parser.add_argument('--gamma', default=0.98, type=float, help='reward gamma')   
        parser.add_argument('--memorysize', default=1000000, type=int, help='memory size')
        parser.add_argument('--batchszie', default=2500, type=int, help='batch size')
        parser.add_argument('--train_iter', default=40000, type=int, help='train iters each timestep')

        args = parser.parse_args()

        #wandb.init(config=args, project="my-project")
        wandb.init(project="my-project")
        wandb.config["test"] = "DQN Version"
        
       
        logs = []
        num_game, self.update_count, ep_reward = 0, 0, 0.
        total_reward, self.total_loss, self.total_q = 0.,0.,0.
        max_avg_ep_reward = 0
        ep_reward, actions = [], []   
        ep_reward_value = 0     
        mean_big = 0
        number_big = 0
        mean_not_big = 0
        number_not_big = 0
       
        ep_target_reward = 0.
        self.env.new_random_game(20)
        for self.step in (range(0, 40000)): # need more configuration
            if self.step == 0:                   # initialize set some varibles
                num_game, self.update_count,ep_reward = 0, 0, 0.
                total_reward, self.total_loss, self.total_q = 0., 0., 0.
                ep_target_reward = 0.
                ep_reward_value = 0     
                ep_reward, actions = [], []          
                
            # prediction
            # action = self.predict(self.history.get())
            if (self.step % 2000 == 1):
                self.env.new_random_game(20) 
            
            print(self.step)
            state_old = self.get_state([0,0])
            #print("state", state_old)
            self.training = True
            for k in range(1):
                
                #i번째 송신 차량
                for i in range(len(self.env.vehicles)):
                    #i번째 송신 차량에서 전송한 신호를 수신하는 j번째 수신 차량 
                    for j in range(3): 
                        # i번째 차량에서 j번째 차량으로 데이터를 전송할때 state를 가져옴.
                        state_old = self.get_state([i,j]) 
                        
                        #state를 보고 action을 정함
                        #action은 선택한 power level, 선택한 resource block 정보를 가짐
                        action = self.predict(state_old, self.step)                    
                        #self.merge_action([i,j], action)   
                        
                        #선택한 resource block을 넣어줌
                        self.action_all_with_power_training[i, j, 0] = action % self.RB_number 
                        
                        
                        
                        #선택한 power level을 넣어줌
                        self.action_all_with_power_training[i, j, 1] = int(np.floor(action/self.RB_number))  
                                         
                        #선택한 power level과 resource block을 기반으로 reward를 계산함.
                        reward_train, reward_best = self.env.act_for_training_HDH(self.action_all_with_power_training, [i,j])

                        ep_reward_value = ep_reward_value + reward_train
                        ep_target_reward = ep_target_reward + reward_best

                        state_new = self.get_state([i,j]) 

                        self.observe(state_old, state_new, reward_train, action)
                        
                        if self.memory.addCount % self.replay_memory_size == 0:
                            self.memory.save('train')
                        
                                 
                logs.append((": 2000 step Cumulative Reward : " + str(ep_reward_value) + '-- Target Reward : ' + str(ep_target_reward) + '-- Diff DQN vs target : ' + str(ep_target_reward - ep_reward_value)))
                wandb.log({"DQN_reward": ep_reward_value, "target_reward" : ep_target_reward, "Diff DQN vs target" : ep_target_reward - ep_reward_value})
                

                ep_target_reward = 0.
                ep_reward_value = 0.

            if (self.step % 2000 == 0) and (self.step > 0):
                # testing 
                self.training = False
                number_of_game = 10
                if (self.step % 10000 == 0) and (self.step > 0):
                    number_of_game = 50 
                if (self.step == 38000):
                    number_of_game = 100               
                V2I_V2X_Rate_list = np.zeros(number_of_game)
                V2I_Rate_list = np.zeros(number_of_game)
                V2V_Rate_list = np.zeros(number_of_game)
                Fail_percent_list = np.zeros(number_of_game)
                
                resourceBlocks = [[0] for _ in range(20)]
                
                
                for game_idx in range(number_of_game):
                    self.env.new_random_game(self.num_vehicle)
                    test_sample = 200
                    Rate_list = []
                    temp_V2V_Rate_list = []
                    temp_V2I_Rate_list = []
                    print('test game idx:', game_idx)
                    for k in range(test_sample):
                        action_temp = self.action_all_with_power.copy()
                        for i in range(len(self.env.vehicles)):
                            self.action_all_with_power[i,:,0] = -1
                            sorted_idx = np.argsort(self.env.individual_time_limit[i,:])          
                            for j in sorted_idx:                   
                                state_old = self.get_state([i,j])
                                action = self.predict(state_old, self.step, True)
                                self.merge_action([i,j], action)
                                resourceBlocks[action % self.RB_number][0] = resourceBlocks[action % self.RB_number][0] + 1
                            if i % (len(self.env.vehicles)/10) == 1:
                                action_temp = self.action_all_with_power.copy()                                
                                returnV2IReward, returnV2VReward, percent = self.env.act_asyn(action_temp) #self.action_all)            
                                Rate_list.append(np.sum(returnV2IReward) + np.sum(returnV2VReward))
                                temp_V2I_Rate_list.append(np.sum(returnV2IReward))
                                temp_V2V_Rate_list.append(np.sum(returnV2VReward))
                        #print("actions", self.action_all_with_power)
                                     
                    V2I_V2X_Rate_list[game_idx] = np.mean(np.asarray(Rate_list))
                    V2I_Rate_list[game_idx] = np.mean(np.asarray(temp_V2I_Rate_list))
                    V2V_Rate_list[game_idx] = np.mean(np.asarray(temp_V2V_Rate_list))
                    Fail_percent_list[game_idx] = percent
                    #print("action is", self.action_all_with_power)
                    print('failure probability is, ', percent)
                    #print('action is that', action_temp[0,:])
                histTable = wandb.Table(data=resourceBlocks, columns=["Resource block"])
                wandb.log({'my_histogram': wandb.plot.histogram(histTable, "Resource block",title="Count of Resource block")})   
                self.save_weight_to_pkl()
                print ('The number of vehicle is ', len(self.env.vehicles))
                print ('Mean of the V2I + V2I rate is that ', np.mean(V2I_V2X_Rate_list))
                print ('Mean of the V2I rate is that ', np.mean(V2I_Rate_list))
                print ('Mean of the V2V rate is that ', np.mean(V2V_Rate_list))
                print('Mean of Fail percent is that ', np.mean(Fail_percent_list))                   
                #print('Test Reward is ', np.mean(test_result))
        self.memory.save('train')    
                  
                    
            
    def q_learning_mini_batch(self):

        # Training the DQN model
        # ------ 
        #s_t, action,reward, s_t_plus_1, terminal = self.memory.sample() 
        s_t, s_t_plus_1, action, reward = self.memory.sample()  
        #print() 
        #print('samples:', s_t[0:10], s_t_plus_1[0:10], action[0:10], reward[0:10])        
        t = time.time()        
        if self.double_q:       #double Q learning   
            pred_action = self.q_action.eval({self.s_t: s_t_plus_1})       
            q_t_plus_1_with_pred_action = self.target_q_with_idx.eval({self.target_s_t: s_t_plus_1, self.target_q_idx: [[idx, pred_a] for idx, pred_a in enumerate(pred_action)]})            
            target_q_t =  self.discount * q_t_plus_1_with_pred_action + reward
        else:
            q_t_plus_1 = self.target_q.eval({self.target_s_t: s_t_plus_1})         
            max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)
            target_q_t = self.discount * max_q_t_plus_1 +reward
        _, q_t, loss,w = self.sess.run([self.optim, self.q, self.loss, self.w], {self.target_q_t: target_q_t, self.action:action, self.s_t:s_t, self.learning_rate_step: self.step}) # training the network
        
        print('loss is ', loss)
        self.total_loss += loss
        self.total_q += q_t.mean()
        self.update_count += 1
            

    def build_dqn(self): 
    # --- Building the DQN -------
        self.w = {}
        self.t_w = {}        
        
        initializer = tf. truncated_normal_initializer(0, 0.02)
        activation_fn = tf.nn.relu
        n_hidden_1 = 500
        n_hidden_2 = 250
        n_hidden_3 = 120
        n_input = 82
        n_output = 60
        def encoder(x):
            weights = {                    
                'encoder_h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1],stddev=0.1)),
                'encoder_h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2],stddev=0.1)),
                'encoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3],stddev=0.1)),
                'encoder_h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_output],stddev=0.1)),
                'encoder_b1': tf.Variable(tf.truncated_normal([n_hidden_1],stddev=0.1)),
                'encoder_b2': tf.Variable(tf.truncated_normal([n_hidden_2],stddev=0.1)),
                'encoder_b3': tf.Variable(tf.truncated_normal([n_hidden_3],stddev=0.1)),
                'encoder_b4': tf.Variable(tf.truncated_normal([n_output],stddev=0.1)),         
            
            }
            layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['encoder_h1']), weights['encoder_b1']))
            layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['encoder_h2']), weights['encoder_b2']))
            layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['encoder_h3']), weights['encoder_b3']))
            layer_4 = tf.nn.relu(tf.add(tf.matmul(layer_3, weights['encoder_h4']), weights['encoder_b4']))
            return layer_4, weights
        with tf.variable_scope('prediction'):
            self.s_t = tf.placeholder('float32',[None, n_input])            
            self.q, self.w = encoder(self.s_t)
            self.q_action = tf.argmax(self.q, dimension = 1)
        with tf.variable_scope('target'):
            self.target_s_t = tf.placeholder('float32', [None, n_input])
            self.target_q, self.target_w = encoder(self.target_s_t)
            self.target_q_idx = tf.placeholder('int32', [None,None], 'output_idx')
            self.target_q_with_idx = tf.gather_nd(self.target_q, self.target_q_idx)
        with tf.variable_scope('pred_to_target'):
            self.t_w_input = {}
            self.t_w_assign_op = {}
            for name in self.w.keys():
                print('name in self w keys', name)
                self.t_w_input[name] = tf.placeholder('float32', self.target_w[name].get_shape().as_list(),name = name)
                self.t_w_assign_op[name] = self.target_w[name].assign(self.t_w_input[name])       
        
        def clipped_error(x):
            try:
                return tf.select(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
            except:
                return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)

        with tf.variable_scope('optimizer'):
            self.target_q_t = tf.placeholder('float32', None, name='target_q_t')
            self.action = tf.placeholder('int32',None, name = 'action')
            action_one_hot = tf.one_hot(self.action, n_output, 1.0, 0.0, name='action_one_hot')
            q_acted = tf.reduce_sum(self.q * action_one_hot, reduction_indices = 1, name='q_acted')
            self.delta = self.target_q_t - q_acted
            self.global_step = tf.Variable(0, trainable=False)
            self.loss = tf.reduce_mean(tf.square(self.delta), name = 'loss')
            self.learning_rate_step = tf.placeholder('int64', None, name='learning_rate_step')
            self.learning_rate_op = tf.maximum(self.learning_rate_minimum, tf.train.exponential_decay(self.learning_rate, self.learning_rate_step, self.learning_rate_decay_step, self.learning_rate_decay, staircase=True))
            self.optim = tf.train.RMSPropOptimizer(self.learning_rate_op, momentum=0.95, epsilon=0.01).minimize(self.loss) 
        
        tf.initialize_all_variables().run()
        self.update_target_q_network()



    def update_target_q_network(self):    
        for name in self.w.keys():
            self.t_w_assign_op[name].eval({self.t_w_input[name]: self.w[name].eval()})       
        
    def save_weight_to_pkl(self): 
        if not os.path.exists(self.weight_dir):
            os.makedirs(self.weight_dir)
        for name in self.w.keys():
            save_pkl(self.w[name].eval(), os.path.join(self.weight_dir,"%s.pkl" % name))       
    def load_weight_from_pkl(self):
        with tf.variable_scope('load_pred_from_pkl'):
            self.w_input = {}
            self.w_assign_op = {}
            for name in self.w.keys():
                self.w_input[name] = tf.placeholder('float32')
                self.w_assign_op[name] = self.w[name].assign(self.w_input[name])
        for name in self.w.keys():
            self.w_assign_op[name].eval({self.w_input[name]:load_pkl(os.path.join(self.weight_dir, "%s.pkl" % name))})
        self.update_target_q_network()   
      
    def play(self, n_step = 100, n_episode = 100, test_ep = None, render = False, random_choice = False):
        number_of_game = n_episode
        self.training = False

        V2I_Rate_list = np.zeros(number_of_game)
        V2V_Rate_list = np.zeros(number_of_game)
        V2V_AvGRate_list = np.zeros(number_of_game)
        Fail_percent_list = np.zeros(number_of_game)
        self.load_weight_from_pkl()
        
        for game_idx in range(number_of_game):
            self.env.new_random_game(self.num_vehicle)
            test_sample = n_step
            Rate_list = []
            Rate_list_V2V = []
            Rate_list_V2VAvg = []
            print('test game idx:', game_idx)
            print('The number of vehicle is ', len(self.env.vehicles))
            
            time_left_list = []
            power_select_list_0 = []
            power_select_list_1 = []
            power_select_list_2 = []

            for k in range(test_sample):
                action_temp = self.action_all_with_power.copy()
                for i in range(len(self.env.vehicles)):
                    self.action_all_with_power[i, :, 0] = -1
                    sorted_idx = np.argsort(self.env.individual_time_limit[i, :])
                    for j in sorted_idx:
                        state_old = self.get_state([i, j])
                        time_left_list.append(state_old[-1])
                        action = self.predict(state_old, 0, True, random_choice = random_choice)
                        
                        if state_old[-1] <=0:
                            continue
                        
                        power_selection = int(np.floor(action/self.RB_number))
                        
                        if power_selection == 0:
                            power_select_list_0.append(state_old[-1])

                        if power_selection == 1:
                            power_select_list_1.append(state_old[-1])
                        if power_selection == 2:
                            power_select_list_2.append(state_old[-1])
                        
                        self.merge_action([i, j], action)
                    
                    #시뮬레이션 차량의 갯수 / 10 만큼 action이 정해지면 act를 수행함.
                    if i % (len(self.env.vehicles) / 10) == 1:
                        action_temp = self.action_all_with_power.copy()
                        rewardOfV2I, rewardOfV2V, percent = self.env.act_asyn(action_temp)  # self.action_all)
                        print('percent : ', percent)
                        Rate_list.append(np.sum(rewardOfV2I))
                        Rate_list_V2V.append(np.sum(rewardOfV2V))
                        Rate_list_V2VAvg.append(np.mean(rewardOfV2V))
                        
                # print("actions", self.action_all_with_power)
            
            
            number_0, bin_edges = np.histogram(power_select_list_0, bins = 10)

            number_1, bin_edges = np.histogram(power_select_list_1, bins = 10)

            number_2, bin_edges = np.histogram(power_select_list_2, bins = 10)


            p_0 = number_0 / (number_0 + number_1 + number_2)
            p_1 = number_1 / (number_0 + number_1 + number_2)
            p_2 = number_2 / (number_0 + number_1 + number_2)

            plt.plot(bin_edges[:-1]*0.1 + 0.01, p_0, 'b*-', label='Power Level 23 dB')
            plt.plot(bin_edges[:-1]*0.1 + 0.01, p_1, 'rs-', label='Power Level 10 dB')
            plt.plot(bin_edges[:-1]*0.1 + 0.01, p_2, 'go-', label='Power Level 5 dB')
            
            plt.xlim([0,0.12])
            plt.xlabel("Time left for V2V transmission (s)")
            plt.ylabel("Probability of power selection")
            plt.legend()
            plt.grid()
            plt.show()
            
            V2I_Rate_list[game_idx] = np.mean(np.asarray(Rate_list))
            V2V_Rate_list[game_idx] = np.mean(np.asarray(Rate_list_V2V))
            V2V_AvGRate_list[game_idx] = np.mean(np.asarray(Rate_list_V2VAvg))
            
            Fail_percent_list[game_idx] = percent

            print('Mean of the V2I rate is that ', np.mean(V2I_Rate_list[0:game_idx] ))
            print('Mean of the V2V rate is that ', np.mean(V2V_Rate_list[0:game_idx] ))
            print('Mean of the patital V2V Avg rate is that ', np.mean(V2V_AvGRate_list[0:game_idx] ))
            print('Mean of Fail percent is that ',percent, np.mean(Fail_percent_list[0:game_idx]))
            # print('action is that', action_temp[0,:])

        print('The number of vehicle is ', len(self.env.vehicles))
        print('Mean of the V2I rate is that ', np.mean(V2I_Rate_list))
        print('Mean of the V2V rate is that ', np.mean(V2V_Rate_list))
        print('Mean of the V2V avg rate is that ', np.mean(V2V_AvGRate_list))
        
        print('Mean of Fail percent is that ', np.mean(Fail_percent_list))
        # print('Test Reward is ', np.mean(test_result))
        
        return np.mean(V2I_Rate_list), np.mean(V2V_Rate_list),np.mean(Fail_percent_list)

    def load_keras_model(self):
        n_hidden_1 = 500
        n_hidden_2 = 250
        n_hidden_3 = 120
        n_input = 82
        n_output = 60


        model= tf.keras.models.Sequential()

        model.add(tf.keras.Input(shape=(n_input,)))
        model.add(tf.keras.layers.Dense(n_hidden_1, activation='relu'))
        model.add(tf.keras.layers.Dense(n_hidden_2, activation='relu'))
        model.add(tf.keras.layers.Dense(n_hidden_3, activation='relu'))
        model.add(tf.keras.layers.Dense(n_output, activation='relu'))
        rootpath = '/home/cnlab1/workspace/MS/5G_V2X_prepare_paper/Codes/3000.3_semester/1.DQN구현체(tensorflow버전)/weight/v2i,v2v,qos,power'
        weight1 = self.loadparameter( os.path.join(rootpath, 'encoder_h1.pkl') ) 
        weight2 = self.loadparameter( os.path.join(rootpath, 'encoder_h2.pkl') ) 
        weight3 = self.loadparameter( os.path.join(rootpath, 'encoder_h3.pkl') ) 
        weight4 = self.loadparameter( os.path.join(rootpath, 'encoder_h4.pkl') ) 
        bias1 = self.loadparameter( os.path.join(rootpath, 'encoder_b1.pkl') ) 
        bias2 = self.loadparameter( os.path.join(rootpath, 'encoder_b2.pkl') ) 
        bias3 = self.loadparameter( os.path.join(rootpath, 'encoder_b3.pkl') ) 
        bias4 = self.loadparameter( os.path.join(rootpath, 'encoder_b4.pkl') ) 

        model.layers[0].set_weights([weight1,bias1])
        model.layers[1].set_weights([weight2,bias2])
        model.layers[2].set_weights([weight3,bias3])
        model.layers[3].set_weights([weight4,bias4])

        return model

    def playwithKeras(self, n_step = 100, n_episode = 100, test_ep = None, render = False, random_choice = False, use_async = True):
        use_async = use_async
        number_of_game = n_episode
        V2I_Rate_list = np.zeros(number_of_game)
        V2V_Rate_list = np.zeros(number_of_game)
        power_list = np.zeros(number_of_game)
        varpower_list = np.zeros(number_of_game)
        stdpower_list = np.zeros(number_of_game)
        V2V_AvGRate_list = np.zeros(number_of_game)
        Fail_percent_list = np.zeros(number_of_game)

        selcted_prob_23dBm = []
        selcted_prob_10dBm = []
        selcted_prob_5dBm = []

        model = self.load_keras_model()
        
        for game_idx in range(number_of_game):
            self.env.new_random_game(self.num_vehicle)
            test_sample = n_step
            Rate_list = []
            Rate_list_V2V = []
            Rate_list_V2VAvg = []
            print('test game idx:', game_idx)
            print('The number of vehicle is ', len(self.env.vehicles))
            
            time_left_list = []
            power_select_list_0 = []
            power_select_list_1 = []
            power_select_list_2 = []

            for k in range(test_sample):
                action_temp = self.action_all_with_power.copy()
                for i in range(len(self.env.vehicles)):
                    self.action_all_with_power[i, :, 0] = -1
                    sorted_idx = np.argsort(self.env.individual_time_limit[i, :])
                    for j in sorted_idx:
                        state_old = self.get_state([i, j])
                        time_left_list.append(state_old[-1])

                        if random_choice == True:
                            action = random.randint(0, 59)
                        else:
                            action = self.predictwidthModel(model, state_old)
                        
                        if state_old[-1] <=0:
                            continue
                        
                        power_selection = int(np.floor(action/self.RB_number))
                        
                        if power_selection == 0:
                            power_select_list_0.append(state_old[-1])

                        if power_selection == 1:
                            power_select_list_1.append(state_old[-1])
                        if power_selection == 2:
                            power_select_list_2.append(state_old[-1])
                        
                        self.merge_action([i, j], action)
                    
                    #시뮬레이션 차량의 갯수 / 10 만큼 action이 정해지면 act를 수행함.
                    if (i % (len(self.env.vehicles) / 10) == 1) & (use_async is True):
                        action_temp = self.action_all_with_power.copy()
                        rewardOfV2I, rewardOfV2V, percent = self.env.act_asyn(action_temp)  # self.action_all)
                        #print('percent : ', percent)
                        Rate_list.append(np.sum(rewardOfV2I))
                        Rate_list_V2V.append(np.sum(rewardOfV2V))
                        Rate_list_V2VAvg.append(np.mean(rewardOfV2V))

                if use_async is False:
                    action_temp = self.action_all_with_power.copy()
                    rewardOfV2I, rewardOfV2V, percent = self.env.act(action_temp) 
                    Rate_list.append(np.sum(rewardOfV2I))
                    Rate_list_V2V.append(np.sum(rewardOfV2V))
                    Rate_list_V2VAvg.append(np.mean(rewardOfV2V))  

                # print("actions", self.action_all_with_power)
            
            selected_power = [200.0 for _ in range(len(power_select_list_0))] + [10.0 for _ in range(len(power_select_list_1))] + [3.2 for _ in range(len(power_select_list_2))]
            # 23dBm -> 200 mW, 10dBm -> 10 mW, 5mW -> 3.2 mW
            
            number_0, bin_edges = np.histogram(power_select_list_0, bins = 10)

            number_1, bin_edges = np.histogram(power_select_list_1, bins = 10)

            number_2, bin_edges = np.histogram(power_select_list_2, bins = 10)


            p_0 = number_0 / (number_0 + number_1 + number_2)
            p_1 = number_1 / (number_0 + number_1 + number_2)
            p_2 = number_2 / (number_0 + number_1 + number_2)

            plt.plot(bin_edges[:-1]*0.1 + 0.01, p_0, 'b*-', label='Power Level 23 dB')
            plt.plot(bin_edges[:-1]*0.1 + 0.01, p_1, 'rs-', label='Power Level 10 dB')
            plt.plot(bin_edges[:-1]*0.1 + 0.01, p_2, 'go-', label='Power Level 5 dB')
            
            plt.xlim([0,0.12])
            plt.xlabel("Time left for V2V transmission (s)")
            plt.ylabel("Probability of power selection")
            plt.legend()
            plt.grid()
            plt.show()

            
            V2I_Rate_list[game_idx] = np.mean(np.asarray(Rate_list))
            V2V_Rate_list[game_idx] = np.mean(np.asarray(Rate_list_V2V))
            V2V_AvGRate_list[game_idx] = np.mean(np.asarray(Rate_list_V2VAvg))

            Fail_percent_list[game_idx] = percent

            power_list[game_idx] = np.mean(selected_power)
            varpower_list[game_idx] = np.var(selected_power)
            stdpower_list[game_idx] = np.std(selected_power)
            selcted_prob_23dBm.append([bin_edges[:-1]*0.1 + 0.01, p_0])
            selcted_prob_10dBm.append([bin_edges[:-1]*0.1 + 0.01, p_1])
            selcted_prob_5dBm.append([bin_edges[:-1]*0.1 + 0.01, p_2])


            print('Mean of the V2I rate is that ', np.mean(V2I_Rate_list[0:game_idx] ))
            print('Mean of the V2V rate is that ', np.mean(V2V_Rate_list[0:game_idx] ))
            print('Mean of the patital V2V Avg rate is that ', np.mean(V2V_AvGRate_list[0:game_idx] ))
            print('Mean of Fail percent is that ',percent, np.mean(Fail_percent_list[0:game_idx]))
            print('Mean of Selected Power ',np.mean(power_list[0:game_idx]))
            print('Mean of Selected var Power ',np.mean(varpower_list[0:game_idx]))
            print('Mean of Selected std Power ',np.mean(stdpower_list[0:game_idx]))
 

            # print('action is that', action_temp[0,:])

        print('The number of vehicle is ', len(self.env.vehicles))
        print('Mean of the V2I rate is that ', np.mean(V2I_Rate_list))
        print('Mean of the V2V rate is that ', np.mean(V2V_Rate_list))
        print('Mean of the V2V avg rate is that ', np.mean(V2V_AvGRate_list))
        
        print('Mean of Fail percent is that ', np.mean(Fail_percent_list))
        print('Mean of Selected Power ',np.mean(power_list))
        print('Mean of Selected var Power ',np.mean(varpower_list))
        print('Mean of Selected std Power ',np.mean(stdpower_list))
        # print('Test Reward is ', np.mean(test_result))
        
        return np.mean(V2I_Rate_list), np.mean(V2V_Rate_list),np.mean(Fail_percent_list), np.mean(power_list),  np.mean(varpower_list),  np.mean(stdpower_list), \
            selcted_prob_23dBm, selcted_prob_10dBm, selcted_prob_5dBm

