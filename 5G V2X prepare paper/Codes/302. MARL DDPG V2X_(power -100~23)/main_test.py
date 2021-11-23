from __future__ import division, print_function
import random
import scipy
import scipy.io
import numpy as np
import tensorflow as tf
import Environment_marl_test
import os
from replay_memory import ReplayMemory
import sys
from main_marl_DDPG_train import Agent as ddpg_agent
from ddpg import DDPG
import argparse

my_config = tf.ConfigProto()
my_config.gpu_options.allow_growth=True

class Agent(object):
    def __init__(self, memory_entry_size):
        self.discount = 1
        self.double_q = True
        self.memory_entry_size = memory_entry_size
        self.memory = ReplayMemory(self.memory_entry_size)

def predict_ddpg(agent, s_t, ep, test_ep = False, decay_epsilon = True):
    pred_action = agent.select_action(s_t, decay_epsilon=decay_epsilon)
    return pred_action

# ################## SETTINGS ######################
up_lanes = [i/2.0 for i in [3.5/2,3.5/2 + 3.5,250+3.5/2, 250+3.5+3.5/2, 500+3.5/2, 500+3.5+3.5/2]]
down_lanes = [i/2.0 for i in [250-3.5-3.5/2,250-3.5/2,500-3.5-3.5/2,500-3.5/2,750-3.5-3.5/2,750-3.5/2]]
left_lanes = [i/2.0 for i in [3.5/2,3.5/2 + 3.5,433+3.5/2, 433+3.5+3.5/2, 866+3.5/2, 866+3.5+3.5/2]]
right_lanes = [i/2.0 for i in [433-3.5-3.5/2,433-3.5/2,866-3.5-3.5/2,866-3.5/2,1299-3.5-3.5/2,1299-3.5/2]]

width = 750/2
height = 1298/2

# This main file is for testing only
IS_TRAIN = 0 # hard-coded to 0
IS_TEST = 1-IS_TRAIN

label = 'marl_model'
label_sarl = 'sarl_model'
label_marl_ddpg ='marl_ddpg_model'

n_veh = 4
n_neighbor = 1
n_RB = n_veh

env = Environment_marl_test.Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height, n_veh, n_neighbor)
env.new_random_game()  # initialize parameters in env

n_episode = 3000
n_step_per_episode = int(env.time_slow/env.time_fast)
epsi_final = 0.02
epsi_anneal_length = int(0.8*n_episode)
mini_batch_step = n_step_per_episode
target_update_step = n_step_per_episode*4

n_episode_test = 100  # test episodes

######################################################


def get_state(env, idx=(0,0), ind_episode=1., epsi=0.02):
    """ Get state from the environment """

    # V2I_channel = (env.V2I_channels_with_fastfading[idx[0], :] - 80) / 60
    V2I_fast = (env.V2I_channels_with_fastfading[idx[0], :] - env.V2I_channels_abs[idx[0]] + 10)/35

    # V2V_channel = (env.V2V_channels_with_fastfading[:, env.vehicles[idx[0]].destinations[idx[1]], :] - 80) / 60
    V2V_fast = (env.V2V_channels_with_fastfading[:, env.vehicles[idx[0]].destinations[idx[1]], :] - env.V2V_channels_abs[:, env.vehicles[idx[0]].destinations[idx[1]]] + 10)/35

    V2V_interference = (-env.V2V_Interference_all[idx[0], idx[1], :] - 60) / 60

    V2I_abs = (env.V2I_channels_abs[idx[0]] - 80) / 60.0
    V2V_abs = (env.V2V_channels_abs[:, env.vehicles[idx[0]].destinations[idx[1]]] - 80)/60.0

    load_remaining = np.asarray([env.demand[idx[0], idx[1]] / env.demand_size])
    time_remaining = np.asarray([env.individual_time_limit[idx[0], idx[1]] / env.time_slow])

    # return np.concatenate((np.reshape(V2V_channel, -1), V2V_interference, V2I_abs, V2V_abs, time_remaining, load_remaining, np.asarray([ind_episode, epsi])))
    return np.concatenate((V2I_fast, np.reshape(V2V_fast, -1), V2V_interference, np.asarray([V2I_abs]), V2V_abs, time_remaining, load_remaining, np.asarray([ind_episode, epsi])))


def get_state_sarl(env, idx=(0,0), ind_episode=1., epsi=0.02):
    """ Get state from the environment """

    # V2I_channel = (env.V2I_channels_with_fastfading[idx[0], :] - 80) / 60
    V2I_fast = (env.V2I_channels_with_fastfading[idx[0], :] - env.V2I_channels_abs[idx[0]] + 10)/35

    # V2V_channel = (env.V2V_channels_with_fastfading[:, env.vehicles[idx[0]].destinations[idx[1]], :] - 80) / 60
    V2V_fast = (env.V2V_channels_with_fastfading[:, env.vehicles[idx[0]].destinations[idx[1]], :] - env.V2V_channels_abs[:, env.vehicles[idx[0]].destinations[idx[1]]] + 10)/35

    V2V_interference = (-env.V2V_Interference_all_sarl[idx[0], idx[1], :] - 60) / 60

    V2I_abs = (env.V2I_channels_abs[idx[0]] - 80) / 60.0
    V2V_abs = (env.V2V_channels_abs[:, env.vehicles[idx[0]].destinations[idx[1]]] - 80)/60.0

    load_remaining = np.asarray([env.demand_sarl[idx[0], idx[1]] / env.demand_size])
    time_remaining = np.asarray([env.individual_time_limit_sarl[idx[0], idx[1]] / env.time_slow])

    # return np.concatenate((np.reshape(V2V_channel, -1), V2V_interference, V2I_abs, V2V_abs, time_remaining, load_remaining, np.asarray([ind_episode, epsi])))
    return np.concatenate((V2I_fast, np.reshape(V2V_fast, -1), V2V_interference, np.asarray([V2I_abs]), V2V_abs, time_remaining, load_remaining, np.asarray([ind_episode, epsi])))

def get_state_ddpg(env, idx=(0,0), ind_episode=1., epsi=0.02):
    """ Get state from the environment """

    # V2I_channel = (env.V2I_channels_with_fastfading[idx[0], :] - 80) / 60
    V2I_fast = (env.V2I_channels_with_fastfading[idx[0], :] - env.V2I_channels_abs[idx[0]] + 10)/35

    # V2V_channel = (env.V2V_channels_with_fastfading[:, env.vehicles[idx[0]].destinations[idx[1]], :] - 80) / 60
    V2V_fast = (env.V2V_channels_with_fastfading[:, env.vehicles[idx[0]].destinations[idx[1]], :] - env.V2V_channels_abs[:, env.vehicles[idx[0]].destinations[idx[1]]] + 10)/35

    V2V_interference = (-env.V2V_Interference_all_ddpg[idx[0], idx[1], :] - 60) / 60

    V2I_abs = (env.V2I_channels_abs[idx[0]] - 80) / 60.0
    V2V_abs = (env.V2V_channels_abs[:, env.vehicles[idx[0]].destinations[idx[1]]] - 80)/60.0

    load_remaining = np.asarray([env.demand_ddpg[idx[0], idx[1]] / env.demand_size])
    time_remaining = np.asarray([env.individual_time_limit_ddpg[idx[0], idx[1]] / env.time_slow])

    # return np.concatenate((np.reshape(V2V_channel, -1), V2V_interference, V2I_abs, V2V_abs, time_remaining, load_remaining, np.asarray([ind_episode, epsi])))
    return np.concatenate((V2I_fast, np.reshape(V2V_fast, -1), V2V_interference, np.asarray([V2I_abs]), V2V_abs, time_remaining, load_remaining, np.asarray([ind_episode, epsi])))


# -----------------------------------------------------------
n_hidden_1 = 500
n_hidden_2 = 250
n_hidden_3 = 120
n_input = len(get_state(env=env))
n_output = n_RB * len(env.V2V_power_dB_List)

g = tf.Graph()
with g.as_default():
    # ============== Training network ========================
    x = tf.placeholder(tf.float32, [None, n_input])

    w_1 = tf.Variable(tf.truncated_normal([n_input, n_hidden_1], stddev=0.1))
    w_2 = tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], stddev=0.1))
    w_3 = tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3], stddev=0.1))
    w_4 = tf.Variable(tf.truncated_normal([n_hidden_3, n_output], stddev=0.1))

    b_1 = tf.Variable(tf.truncated_normal([n_hidden_1], stddev=0.1))
    b_2 = tf.Variable(tf.truncated_normal([n_hidden_2], stddev=0.1))
    b_3 = tf.Variable(tf.truncated_normal([n_hidden_3], stddev=0.1))
    b_4 = tf.Variable(tf.truncated_normal([n_output], stddev=0.1))

    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, w_1), b_1))
    layer_1_b = tf.layers.batch_normalization(layer_1)
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1_b, w_2), b_2))
    layer_2_b = tf.layers.batch_normalization(layer_2)
    layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2_b, w_3), b_3))
    layer_3_b = tf.layers.batch_normalization(layer_3)
    y = tf.nn.relu(tf.add(tf.matmul(layer_3, w_4), b_4))
    g_q_action = tf.argmax(y, axis=1)

    # compute loss
    g_target_q_t = tf.placeholder(tf.float32, None, name="target_value")
    g_action = tf.placeholder(tf.int32, None, name='g_action')
    action_one_hot = tf.one_hot(g_action, n_output, 1.0, 0.0, name='action_one_hot')
    q_acted = tf.reduce_sum(y * action_one_hot, reduction_indices=1, name='q_acted')

    g_loss = tf.reduce_mean(tf.square(g_target_q_t - q_acted), name='g_loss')
    optim = tf.train.RMSPropOptimizer(learning_rate=0.001, momentum=0.95, epsilon=0.01).minimize(g_loss)

    # ==================== Prediction network ========================
    x_p = tf.placeholder(tf.float32, [None, n_input])

    w_1_p = tf.Variable(tf.truncated_normal([n_input, n_hidden_1], stddev=0.1))
    w_2_p = tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], stddev=0.1))
    w_3_p = tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3], stddev=0.1))
    w_4_p = tf.Variable(tf.truncated_normal([n_hidden_3, n_output], stddev=0.1))

    b_1_p = tf.Variable(tf.truncated_normal([n_hidden_1], stddev=0.1))
    b_2_p = tf.Variable(tf.truncated_normal([n_hidden_2], stddev=0.1))
    b_3_p = tf.Variable(tf.truncated_normal([n_hidden_3], stddev=0.1))
    b_4_p = tf.Variable(tf.truncated_normal([n_output], stddev=0.1))

    layer_1_p = tf.nn.relu(tf.add(tf.matmul(x_p, w_1_p), b_1_p))
    layer_1_p_b = tf.layers.batch_normalization(layer_1_p)

    layer_2_p = tf.nn.relu(tf.add(tf.matmul(layer_1_p_b, w_2_p), b_2_p))
    layer_2_p_b = tf.layers.batch_normalization(layer_2_p)

    layer_3_p = tf.nn.relu(tf.add(tf.matmul(layer_2_p_b, w_3_p), b_3_p))
    layer_3_p_b = tf.layers.batch_normalization(layer_3_p)

    y_p = tf.nn.relu(tf.add(tf.matmul(layer_3_p_b, w_4_p), b_4_p))

    g_target_q_idx = tf.placeholder('int32', [None, None], 'output_idx')
    target_q_with_idx = tf.gather_nd(y_p, g_target_q_idx)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()


def predict(sess, s_t, ep, test_ep = False):

    n_power_levels = len(env.V2V_power_dB_List)
    if np.random.rand() < ep and not test_ep:
        pred_action = np.random.randint(n_RB*n_power_levels)
    else:
        pred_action = sess.run(g_q_action, feed_dict={x: [s_t]})[0]
    return pred_action

def predict_sarl(sess, s_t):
    pred_action = sess.run(g_q_action, feed_dict={x: [s_t]})[0]
    return pred_action


def q_learning_mini_batch(current_agent, current_sess):
    """ Training a sampled mini-batch """

    batch_s_t, batch_s_t_plus_1, batch_action, batch_reward = current_agent.memory.sample()

    if current_agent.double_q:  # double q-learning
        pred_action = current_sess.run(g_q_action, feed_dict={x: batch_s_t_plus_1})
        q_t_plus_1 = current_sess.run(target_q_with_idx, {x_p: batch_s_t_plus_1, g_target_q_idx: [[idx, pred_a] for idx, pred_a in enumerate(pred_action)]})
        batch_target_q_t = current_agent.discount * q_t_plus_1 + batch_reward
    else:
        q_t_plus_1 = current_sess.run(y_p, {x_p: batch_s_t_plus_1})
        max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)
        batch_target_q_t = current_agent.discount * max_q_t_plus_1 + batch_reward

    _, loss_val = current_sess.run([optim, g_loss], {g_target_q_t: batch_target_q_t, g_action: batch_action, x: batch_s_t})
    return loss_val


def update_target_q_network(sess):
    """ Update target q network once in a while """

    sess.run(w_1_p.assign(sess.run(w_1)))
    sess.run(w_2_p.assign(sess.run(w_2)))
    sess.run(w_3_p.assign(sess.run(w_3)))
    sess.run(w_4_p.assign(sess.run(w_4)))

    sess.run(b_1_p.assign(sess.run(b_1)))
    sess.run(b_2_p.assign(sess.run(b_2)))
    sess.run(b_3_p.assign(sess.run(b_3)))
    sess.run(b_4_p.assign(sess.run(b_4)))


def save_models(sess, model_path):
    """ Save models to the current directory with the name filename """

    current_dir = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(current_dir, "model/" + model_path)
    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))
    saver.save(sess, model_path, write_meta_graph=False)


    
    
def load_models(sess, model_path):
    """ Restore models from the current directory with the name filename """

    dir_ = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(dir_, "model/" + model_path)
    saver.restore(sess, model_path)


def print_weight(sess, target=False):
    """ debug """

    if not target:
        print(sess.run(w_1[0, 0:4]))
    else:
        print(sess.run(w_1_p[0, 0:4]))


parser = argparse.ArgumentParser(description='PyTorch on TORCS with Multi-modal')

parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
parser.add_argument('--hidden1', default=512, type=int, help='hidden num of first fully connect layer')
parser.add_argument('--hidden2', default=256, type=int, help='hidden num of second fully connect layer')
parser.add_argument('--rate', default=0.001, type=float, help='learning rate')
parser.add_argument('--prate', default=0.0001, type=float, help='policy net learning rate (only for DDPG)')
    
parser.add_argument('--discount', default=0.99, type=float, help='')
parser.add_argument('--bsize', default=64, type=int, help='minibatch size') # 64
parser.add_argument('--rmsize', default=6000000, type=int, help='memory size')
parser.add_argument('--window_length', default=1, type=int, help='')
parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
parser.add_argument('--ou_theta', default=0.15, type=float, help='noise theta')
parser.add_argument('--ou_sigma', default=0.2, type=float, help='noise sigma') 
parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu') 
parser.add_argument('--validate_episodes', default=20, type=int, help='how many episode to perform during validate experiment')
parser.add_argument('--max_episode_length', default=500, type=int, help='')

parser.add_argument('--output', default='output', type=str, help='')
parser.add_argument('--debug', dest='debug', action='store_true')
    
parser.add_argument('--init_w', default=0.003, type=float, help='') 
parser.add_argument('--warmup', default=10000, type=int, help='time without training but only filling the replay memory') # 10000
parser.add_argument('--validate_steps', default=2000, type=int, help='how many steps to perform a validate experiment') # 2000
parser.add_argument('--train_iter', default=20000, type=int, help='train iters each timestep') # 200000
parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
parser.add_argument('--seed', default=-1, type=int, help='')
parser.add_argument('--resume', default='default', type=str, help='Resuming model path for testing')
    
nb_states = len(get_state(env=env))
nb_actions = 2

args = parser.parse_args()
ddpg_agents = []
ddpg_actor_path = ['./model/marl_ddpg_model/agent_0/actor_0_9999.pkl',
                   './model/marl_ddpg_model/agent_1/actor_1_9999.pkl',
                   './model/marl_ddpg_model/agent_2/actor_2_9999.pkl',
                   './model/marl_ddpg_model/agent_3/actor_3_9999.pkl',]

ddpg_critic_path = ['./model/marl_ddpg_model/agent_0/critic_0_9999.pkl',
                   './model/marl_ddpg_model/agent_1/critic_1_9999.pkl',
                   './model/marl_ddpg_model/agent_2/critic_2_9999.pkl',
                   './model/marl_ddpg_model/agent_3/critic_3_9999.pkl',]

for ind_agent in range(n_veh * n_neighbor):  # initialize agents   
    DDPGModel = DDPG(nb_states, nb_actions, args, n_veh)
    tempagent = ddpg_agent(DDPGModel)
    ddpg_agents.append(tempagent)
    ddpg_agents[len(ddpg_agents)-1].load_model(ddpg_actor_path[ind_agent], ddpg_critic_path[ind_agent])

# --------------------------------------------------------------
agents = []
sesses = []
for ind_agent in range(n_veh * n_neighbor):  # initialize agents
    print("Initializing agent", ind_agent)
    agent = Agent(memory_entry_size=len(get_state(env)))
    agents.append(agent)

    sess = tf.Session(graph=g,config=my_config)
    sess.run(init)
    sesses.append(sess)

agent_sarl = Agent(memory_entry_size=len(get_state(env)))
sess_sarl = tf.Session(graph=g,config=my_config)
sess_sarl.run(init)

# -------------- Testing --------------
if IS_TEST:
    print("\nRestoring the model...")

    for i in range(n_veh):
        for j in range(n_neighbor):
            model_path = label + '/agent_' + str(i * n_neighbor + j)
            load_models(sesses[i * n_neighbor + j], model_path)
    # restore the single-agent model
    model_path_single = label_sarl + '/agent'
    load_models(sess_sarl, model_path_single)

    V2I_rate_list = []
    V2V_success_list = []

    V2I_rate_list_ddpg = []
    V2V_success_list_ddpg = []

    V2I_rate_list_rand = []
    V2V_success_list_rand = []

    V2I_rate_list_sarl = []
    V2V_success_list_sarl = []

    V2I_rate_list_dpra = []
    V2V_success_list_dpra = []

    rate_marl = np.zeros([n_episode_test, n_step_per_episode, n_veh, n_neighbor])
    rate_ddpg = np.zeros([n_episode_test, n_step_per_episode, n_veh, n_neighbor])
    rate_rand = np.zeros([n_episode_test, n_step_per_episode, n_veh, n_neighbor])
    
    demand_marl = env.demand_size * np.ones([n_episode_test, n_step_per_episode+1, n_veh, n_neighbor])
    demand_ddpg = env.demand_size * np.ones([n_episode_test, n_step_per_episode+1, n_veh, n_neighbor])
    demand_rand = env.demand_size * np.ones([n_episode_test, n_step_per_episode+1, n_veh, n_neighbor])

    action_all_testing_ddpg = np.zeros([n_veh, n_neighbor, 2], dtype='float')
    action_all_testing_sarl = np.zeros([n_veh, n_neighbor, 2], dtype='int32')
    action_all_testing_dpra = np.zeros([n_veh, n_neighbor, 2], dtype='int32')
    
    for idx_episode in range(n_episode_test):
        print('----- Episode', idx_episode, '-----')

        env.renew_positions()
        env.renew_neighbor()
        env.renew_channel()
        env.renew_channels_fastfading()

        env.demand = env.demand_size * np.ones((env.n_Veh, env.n_neighbor))
        env.individual_time_limit = env.time_slow * np.ones((env.n_Veh, env.n_neighbor))
        env.active_links = np.ones((env.n_Veh, env.n_neighbor), dtype='bool')

        env.demand_ddpg = env.demand_size * np.ones((env.n_Veh, env.n_neighbor))
        env.individual_time_limit_ddpg = env.time_slow * np.ones((env.n_Veh, env.n_neighbor))
        env.active_links_ddpg = np.ones((env.n_Veh, env.n_neighbor), dtype='bool')

        env.demand_rand = env.demand_size * np.ones((env.n_Veh, env.n_neighbor))
        env.individual_time_limit_rand = env.time_slow * np.ones((env.n_Veh, env.n_neighbor))
        env.active_links_rand = np.ones((env.n_Veh, env.n_neighbor), dtype='bool')

        env.demand_sarl = env.demand_size * np.ones((env.n_Veh, env.n_neighbor))
        env.individual_time_limit_sarl = env.time_slow * np.ones((env.n_Veh, env.n_neighbor))
        env.active_links_sarl = np.ones((env.n_Veh, env.n_neighbor), dtype='bool')

        env.demand_dpra = env.demand_size * np.ones((env.n_Veh, env.n_neighbor))
        env.individual_time_limit_dpra = env.time_slow * np.ones((env.n_Veh, env.n_neighbor))
        env.active_links_dpra = np.ones((env.n_Veh, env.n_neighbor), dtype='bool')

        V2I_rate_per_episode = []
        V2I_rate_per_episode_ddpg = []
        V2I_rate_per_episode_rand = []
        V2I_rate_per_episode_sarl = []
        V2I_rate_per_episode_dpra = []

        for test_step in range(n_step_per_episode):
            
            # trained models
            action_all_testing = np.zeros([n_veh, n_neighbor, 2], dtype='int32')
            for i in range(n_veh):
                for j in range(n_neighbor):
                    state_old = get_state(env, [i, j], 1, epsi_final)
                    action = predict(sesses[i*n_neighbor+j], state_old, epsi_final, True)
                    action_all_testing[i, j, 0] = action % n_RB  # chosen RB
                    action_all_testing[i, j, 1] = int(np.floor(action / n_RB))  # power level

            action_temp = action_all_testing.copy()
            V2I_rate, V2V_success, V2V_rate = env.act_for_testing(action_temp)
            V2I_rate_per_episode.append(np.sum(V2I_rate))  # sum V2I rate in bps

            rate_marl[idx_episode, test_step,:,:] = V2V_rate
            demand_marl[idx_episode, test_step+1,:,:] = env.demand

            # trained DDPG models
            action_all_testing_ddpg = np.zeros([n_veh, n_neighbor, 2], dtype='float')
            
            for i in range(n_veh):
                for j in range(n_neighbor):
                    state_old = get_state_ddpg(env, [i, j], 1, epsi_final)
                    action =  predict_ddpg(ddpg_agents[i*n_neighbor+j], state_old, ep = epsi_final ,test_ep = True,decay_epsilon = False )
                    action_all_testing_ddpg[i, j, 0] = action[0]
                    action_all_testing_ddpg[i, j, 1] = action[1]
                    
            
            action_temp_ddpg = action_all_testing_ddpg.copy()
            V2I_rate_ddpg, V2V_success_ddpg, V2V_rate_ddpg = env.act_for_testing_ddpg(action_temp_ddpg)
            V2I_rate_per_episode_ddpg.append(np.sum(V2I_rate_ddpg))  # sum V2I rate in bps
            
            rate_ddpg[idx_episode, test_step,:,:] = V2V_rate_ddpg
            demand_ddpg[idx_episode, test_step+1,:,:] = env.demand_ddpg
            
            
            # random baseline
            action_rand = np.zeros([n_veh, n_neighbor, 2], dtype='int32')
            action_rand[:, :, 0] = np.random.randint(0, n_RB, [n_veh, n_neighbor]) # band
            action_rand[:, :, 1] = np.random.randint(0, len(env.V2V_power_dB_List), [n_veh, n_neighbor]) # power
            V2I_rate_rand, V2V_success_rand, V2V_rate_rand = env.act_for_testing_rand(action_rand)
            V2I_rate_per_episode_rand.append(np.sum(V2I_rate_rand))  # sum V2I rate in bps

            rate_rand[idx_episode, test_step, :, :] = V2V_rate_rand
            demand_rand[idx_episode, test_step+1,:,:] = env.demand_rand
            

            # SARL
            remainder = test_step % (n_veh * n_neighbor)
            i = int(np.floor(remainder/n_neighbor))
            j = remainder % n_neighbor
            state_sarl = get_state_sarl(env, [i, j], 1, epsi_final)
            action = predict_sarl(sess_sarl, state_sarl)
            action_all_testing_sarl[i, j, 0] = action % n_RB  # chosen RB
            action_all_testing_sarl[i, j, 1] = int(np.floor(action / n_RB))  # power level
            action_temp_sarl = action_all_testing_sarl.copy()
            V2I_rate_sarl, V2V_success_sarl, V2V_rate_sarl = env.act_for_testing_sarl(action_temp_sarl)
            V2I_rate_per_episode_sarl.append(np.sum(V2I_rate_sarl))  # sum V2I rate in bps

            # # Used as V2I upper bound only, no V2V transmission
            # action_all_testing_dpra[i, j, 0] = 0  # chosen RB
            # action_all_testing_dpra[i, j, 1] = 3  # power level, fixed to -100 dBm, no V2V transmission
            #
            # action_temp_dpra = action_all_testing_dpra.copy()
            # V2I_rate_dpra, V2V_success_dpra, V2V_rate_dpra = env.act_for_testing_dpra(action_temp_dpra)
            # V2I_rate_per_episode_dpra.append(np.sum(V2I_rate_dpra))  # sum V2I rate in bps

            # # V2V Upper bound only, centralized maxV2V
            # The following applies to n_veh = 4 and n_neighbor = 1 only
            action_dpra = np.zeros([n_veh, n_neighbor, 2], dtype='int32')
            # n_power_level = len(env.V2V_power_dB_List)
            n_power_level = 1
            store_action = np.zeros([(n_RB*n_power_level)**4, 4])
            rate_all_dpra = []
            t = 0
            # for i in range(n_RB*len(env.V2V_power_dB_List)):\
            for i in range(n_RB):
                for j in range(n_RB):
                    for m in range(n_RB):
                        for n in range(n_RB):
                            action_dpra[0, 0, 0] = i % n_RB
                            action_dpra[0, 0, 1] = int(np.floor(i / n_RB))  # power level

                            action_dpra[1, 0, 0] = j % n_RB
                            action_dpra[1, 0, 1] = int(np.floor(j / n_RB))  # power level

                            action_dpra[2, 0, 0] = m % n_RB
                            action_dpra[2, 0, 1] = int(np.floor(m / n_RB))  # power level

                            action_dpra[3, 0, 0] = n % n_RB
                            action_dpra[3, 0, 1] = int(np.floor(n / n_RB))  # power level

                            action_temp_findMax = action_dpra.copy()
                            V2I_rate_findMax, V2V_rate_findMax = env.Compute_Rate(action_temp_findMax)
                            rate_all_dpra.append(np.sum(V2V_rate_findMax))

                            store_action[t, :] = [i,j,m,n]
                            t += 1

            i = store_action[np.argmax(rate_all_dpra), 0]
            j = store_action[np.argmax(rate_all_dpra), 1]
            m = store_action[np.argmax(rate_all_dpra), 2]
            n = store_action[np.argmax(rate_all_dpra), 3]

            action_testing_dpra = np.zeros([n_veh, n_neighbor, 2], dtype='int32')

            action_testing_dpra[0, 0, 0] = i % n_RB
            action_testing_dpra[0, 0, 1] = int(np.floor(i / n_RB))  # power level

            action_testing_dpra[1, 0, 0] = j % n_RB
            action_testing_dpra[1, 0, 1] = int(np.floor(j / n_RB))  # power level

            action_testing_dpra[2, 0, 0] = m % n_RB
            action_testing_dpra[2, 0, 1] = int(np.floor(m / n_RB))  # power level

            action_testing_dpra[3, 0, 0] = n % n_RB
            action_testing_dpra[3, 0, 1] = int(np.floor(n / n_RB))  # power level

            V2I_rate_findMax, V2V_rate_findMax = env.Compute_Rate(action_testing_dpra)
            check_sum = np.sum(V2V_rate_findMax)

            action_temp_dpra = action_testing_dpra.copy()
            V2I_rate_dpra, V2V_success_dpra, V2V_rate_dpra = env.act_for_testing_dpra(action_temp_dpra)
            V2I_rate_per_episode_dpra.append(np.sum(V2I_rate_dpra))  # sum V2I rate in bps

            # update the environment and compute interference
            env.renew_channels_fastfading()
            env.Compute_Interference(action_temp)
            env.Compute_Interference_ddpg(action_temp_ddpg)
            env.Compute_Interference_sarl(action_temp_sarl)
            env.Compute_Interference_dpra(action_temp_dpra)

            if test_step == n_step_per_episode - 1:
                V2V_success_list.append(V2V_success)
                V2V_success_list_rand.append(V2V_success_rand)
                V2V_success_list_sarl.append(V2V_success_sarl)
                V2V_success_list_dpra.append(V2V_success_dpra)
                V2V_success_list_ddpg.append(V2V_success_ddpg)
                
            
        V2I_rate_list.append(np.mean(V2I_rate_per_episode))
        V2I_rate_list_rand.append(np.mean(V2I_rate_per_episode_rand))
        V2I_rate_list_sarl.append(np.mean(V2I_rate_per_episode_sarl))
        V2I_rate_list_dpra.append(np.mean(V2I_rate_per_episode_dpra))
        V2I_rate_list_ddpg.append(np.mean(V2I_rate_per_episode_ddpg))  
        
        print('ddpg', round(np.average(V2I_rate_per_episode_ddpg), 2),'marl', round(np.average(V2I_rate_per_episode), 2), 'sarl', round(np.average(V2I_rate_per_episode_sarl), 2), 'dpra', round(np.average(V2I_rate_per_episode_dpra), 2))
        print('ddpg', V2V_success_list_ddpg[idx_episode],'marl', V2V_success_list[idx_episode], 'sarl', V2V_success_list_sarl[idx_episode], 'dpra', V2V_success_list_dpra[idx_episode])

    print('-------- ddpg -------------')
    print('n_veh:', n_veh, ', n_neighbor:', n_neighbor)
    print('Sum V2I rate:', round(np.average(V2I_rate_list_ddpg), 2), 'Mbps')
    print('Pr(V2V success):', round(np.average(V2V_success_list_ddpg), 4))
    
    print('-------- marl -------------')
    print('n_veh:', n_veh, ', n_neighbor:', n_neighbor)
    print('Sum V2I rate:', round(np.average(V2I_rate_list), 2), 'Mbps')
    print('Pr(V2V success):', round(np.average(V2V_success_list), 4))
    #
    print('-------- sarl -------------')
    print('n_veh:', n_veh, ', n_neighbor:', n_neighbor)
    print('Sum V2I rate:', round(np.average(V2I_rate_list_sarl), 2), 'Mbps')
    print('Pr(V2V success):', round(np.average(V2V_success_list_sarl), 4))
    """
    print('-------- random -------------')
    print('n_veh:', n_veh, ', n_neighbor:', n_neighbor)
    print('Sum V2I rate:', round(np.average(V2I_rate_list_rand), 2), 'Mbps')
    print('Pr(V2V success):', round(np.average(V2V_success_list_rand), 4))
    """
    print('-------- DPRA -------------')
    print('n_veh:', n_veh, ', n_neighbor:', n_neighbor)
    print('Sum V2I rate:', round(np.average(V2I_rate_list_dpra), 2), 'Mbps')
    print('Pr(V2V success):', round(np.average(V2V_success_list_dpra), 4))

# The name "DPRA" is used for historical reasons. Not really the case...

    with open("Data.txt", "a") as f:
        f.write('-------- ddpg, ' + label + '------\n')
        f.write('n_veh: ' + str(n_veh) + ', n_neighbor: ' + str(n_neighbor) + '\n')
        f.write('Sum V2I rate: ' + str(round(np.average(V2I_rate_list_ddpg), 5)) + ' Mbps\n')
        f.write('Pr(V2V): ' + str(round(np.average(V2V_success_list_ddpg), 5)) + '\n')       
        f.write('-------- marl, ' + label + '------\n')
        f.write('n_veh: ' + str(n_veh) + ', n_neighbor: ' + str(n_neighbor) + '\n')
        f.write('Sum V2I rate: ' + str(round(np.average(V2I_rate_list), 5)) + ' Mbps\n')
        f.write('Pr(V2V): ' + str(round(np.average(V2V_success_list), 5)) + '\n')
        f.write('-------- sarl, ' + label_sarl + '------\n')
        f.write('n_veh: ' + str(n_veh) + ', n_neighbor: ' + str(n_neighbor) + '\n')
        f.write('Sum V2I rate: ' + str(round(np.average(V2I_rate_list_sarl), 5)) + ' Mbps\n')
        f.write('Pr(V2V): ' + str(round(np.average(V2V_success_list_sarl), 5)) + '\n')
        f.write('--------random ------------\n')
        f.write('Rand Sum V2I rate: ' + str(round(np.average(V2I_rate_list_rand), 5)) + ' Mbps\n')
        f.write('Rand Pr(V2V): ' + str(round(np.average(V2V_success_list_rand), 5)) + '\n')
        f.write('--------DPRA ------------\n')
        f.write('Dpra Sum V2I rate: ' + str(round(np.average(V2I_rate_list_dpra), 5)) + ' Mbps\n')
        f.write('Dpra Pr(V2V): ' + str(round(np.average(V2V_success_list_dpra), 5)) + '\n')

    current_dir = os.path.dirname(os.path.realpath(__file__))
    marl_path = os.path.join(current_dir, "model/" + label + '/rate_marl.mat')
    scipy.io.savemat(marl_path, {'rate_marl': rate_marl})
    rand_path = os.path.join(current_dir, "model/" + label + '/rate_rand.mat')
    scipy.io.savemat(rand_path, {'rate_rand': rate_rand})

    demand_marl_path = os.path.join(current_dir, "model/" + label + '/demand_marl.mat')
    scipy.io.savemat(demand_marl_path, {'demand_marl': demand_marl})
    demand_rand_path = os.path.join(current_dir, "model/" + label + '/demand_rand.mat')
    scipy.io.savemat(demand_rand_path, {'demand_rand': demand_rand})


# close sessions
for sess in sesses:
    sess.close()


# if __name__ == '__main__':
#     tf.app.run()