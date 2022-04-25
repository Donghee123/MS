import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.reset_default_graph() # in the begining of the code to avoid namespace error

# to be able to compare methods it should be set after tf.reset_default_graph()
tf.set_random_seed(1)
np.random.seed(1)


# set initial parameters
num_hidden = 20
num_steps = 10  # as we want to read the images column by column
num_ant = 100 # determines number of inputs
num_classes = 2
num_epochs = 100
batch_size = 32
phase_noise_coef = 1
data_type = 'phase_only' # channels, phase_only, amplitude_only    three ways of using data 1) channels use the channel real and imag part (which are in the same order), 2)phase_only just use the phase and ignore abs value of channel, 3) amplitude just use the channel amplitude and ignores phase
##################### Load the data #######################
def import_data(data_type):
    import scipy.io as sio
    InputDict_train = sio.loadmat('/home/meysam/Downloads/Lund/COST model/cost_model/MEYSAM/train_snapnum10_snaprate4_ant100.mat')
    InputDict_test = sio.loadmat('/home/meysam/Downloads/Lund/COST model/cost_model/MEYSAM/test_snapnum10_snaprate4_ant100.mat')
    if data_type == 'channels':
        l2r_train  = InputDict_train['LeftToRight'].reshape([-1, num_steps, num_ant * 2])
        r2l_train  = InputDict_train['RightToLeft'].reshape([-1, num_steps, num_ant * 2])
        l2r_test   = InputDict_test['LeftToRight'].reshape([-1, num_steps, num_ant * 2])
        r2l_test   = InputDict_test['RightToLeft'].reshape([-1, num_steps, num_ant * 2])
    elif data_type == 'phase_only':
        l2r_train = InputDict_train['LeftToRight_phase_amp'][:,:,:,0] # just take the phase info
        r2l_train = InputDict_train['RightToLeft_phase_amp'][:,:,:,0] # just take the phase info
        # sanity test for phase case np.cos(X_train[a,b,c]) = X_train[a,b,2*c] / np.sqrt(X_train[a,b,2*c]**2 + X_train[a,b,2*c+1]**2)
        l2r_test = InputDict_test['LeftToRight_phase_amp'][:,:,:,0]
        r2l_test = InputDict_test['RightToLeft_phase_amp'][:,:,:,0]
        
    elif data_type == 'amplitude_only':
        l2r_train = InputDict_train['LeftToRight_phase_amp'][:,:,:,1] # just take the phase info
        r2l_train = InputDict_train['RightToLeft_phase_amp'][:,:,:,1] # just take the phase info
        l2r_test  = InputDict_test['LeftToRight_phase_amp'][:,:,:,1]
        r2l_test  = InputDict_test['RightToLeft_phase_amp'][:,:,:,1]
        
    X_train = np.vstack((l2r_train,r2l_train))  # num_sample x num_time_steps x num_BS_ant x 2 (2 is for real and imaginary parts of channel)
    Y_train = np.vstack( ( np.hstack((np.ones([len(l2r_train)]), np.zeros([len(r2l_train)])))  , np.hstack((np.zeros([len(l2r_train)]), np.ones([len(r2l_train)])))  ) ).T
    X_test = np.vstack((l2r_test,r2l_test))  # num_sample x num_time_steps x num_BS_ant x 2 (2 is for real and imaginary parts of channel)
    Y_test = np.vstack( ( np.hstack((np.ones([len(l2r_test)]), np.zeros([len(r2l_test)])))  , np.hstack((np.zeros([len(l2r_test)]), np.ones([len(r2l_test)])))  ) ).T    
    return X_train, Y_train, X_test, Y_test
    
        
            
X_train, Y_train, X_test, Y_test = import_data(data_type)  
random_noise = np.random.uniform(low=0.0, high=6.28, size=X_test.shape)
X_test_noisy = X_test + (phase_noise_coef * random_noise)
############################################################
num_batches = int(len(Y_train) / batch_size)

############################################################
def NextBatch(X,Y,batch_size):
    idx = np.arange(len(Y))
    np.random.shuffle(idx)
    X_batch = np.array([X[i] for i in idx[:batch_size]])
    Y_batch = np.array([Y[i] for i in idx[:batch_size]])
    return X_batch, Y_batch


############################################################
# PLaceholders
Y = tf.placeholder(tf.float32,[None,num_classes])
if data_type == 'channels':
    X = tf.placeholder(tf.float32, [None, num_steps, num_ant * 2])
else:
    X = tf.placeholder(tf.float32, [None, num_steps, num_ant])
############################################################
# Create the RNN
def RNN_model(x,num_classes): 
    # Define an lstm cell with tensorflow
    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units = num_hidden,state_is_tuple=True) #lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Get lstm cell output
    #outputs: is the RNN output tensor. If time_major == False (default), this will be a Tensor shaped: [batch_size, max_time, cell.output_size] = [batch_size, num_time_steps, num_hidden_units]
    #last_states:is the final state of RNN.  cell.state_size is an int, this will be shaped [batch_size, cell.state_size]. If it is a TensorShape, this will be shaped [batch_size] + cell.state_size
    
    outputs, last_states = tf.nn.dynamic_rnn(cell=lstm_cell,  # an instance of RNN cell
                                             inputs=x,        # The RNN inputs. If time_major == False (default), this must be a Tensor of shape: [batch_size, max_time, ...], or a nested tuple of such elements
                                             dtype=tf.float32 # It is NOT optional, if we do not provide 
                                             # sequence_length = sequence_length # this one is optional (read the note above on sequence_length). When all our input data points have the same number of time steps
                                             # time_major = False # It is optional. time_major determines the shape format of the inputs and outputs Tensors. If true, these Tensors must be shaped [max_time, batch_size, depth]. If false, these Tensors must be shaped [batch_size, max_time, depth].
                                            )
  
    # If you do not need batch normalization, comment next line and change the return
    batch_normzd = tf.layers.batch_normalization(outputs[:, -1, :])
    
    y_hat = tf.layers.dense(batch_normzd,num_classes, activation=None, kernel_initializer=tf.orthogonal_initializer())
    return y_hat
    





############################################################
logits = RNN_model(X,num_classes)
predictions = tf.nn.softmax(logits)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y,logits=logits))
optimizer = tf.train.AdamOptimizer().minimize(cost)

accuracy = tf.reduce_mean( (tf.cast(tf.equal(tf.argmax(predictions,1), tf.argmax(Y,1)),dtype=tf.float32)) )

#saver = tf.train.Saver()

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch_cntr in range(num_epochs):
        for batch_cntr in range(num_batches):
            x_train_batch, y_train_batch = NextBatch(X_train,Y_train,batch_size)
            sess.run(optimizer, feed_dict={X:x_train_batch, Y:y_train_batch})
            batch_train_cost,batch_train_acc= sess.run([cost,accuracy], feed_dict={X:x_train_batch, Y:y_train_batch})
            
            if batch_cntr  % (num_batches) == 0:# % num_batches == 0:
                print('batch_cntr',batch_cntr)
                print('epoch_cntr',epoch_cntr)
                x_test_batch, y_test_batch = NextBatch(X_test_noisy,Y_test,1024)  # NextBatch(X_test_noisy,Y_test,1024)
                batch_test_cost,batch_test_acc= sess.run([cost,accuracy], feed_dict={X:x_test_batch, Y:y_test_batch})
                print('\n Train Acc:{}   Test Acc:{}    Train Cost:{}   Test Cost:{}'.format(batch_train_acc, batch_test_acc, batch_train_cost, batch_test_cost))
                #print('\n Train Acc:{}   Train Cost:{}'.format(batch_train_acc, batch_train_cost))
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                