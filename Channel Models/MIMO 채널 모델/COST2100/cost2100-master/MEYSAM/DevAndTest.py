# developement and test

# Keras Implementation

from keras.models import Sequential
from keras.layers import LSTM, Dense, BatchNormalization
import numpy as np
from keras import backend as K
import matplotlib.pyplot as plt


# NN initial parameters
batch_size = 512
num_hidden = 20
num_steps = 10  # 
num_classes = 2
data_dim = 100 # num_ant determines number of inputs
timesteps = 10 # num_steps
num_units = 20 # determine the size of activation vector, also determine the Waa which will be num_units*num_units
num_classes = 2
# Data Initial Parameters
num_ant = 100 # determines number of inputs
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
# random shuffle of data
idx = np.arange(X_train.shape[0])
np.random.shuffle(idx)       
X_train = X_train[idx]
Y_train = Y_train[idx]

print('X_train.shape', X_train.shape)
print('Y_train.shape', Y_train.shape)
# add phase noise
random_noise = np.random.uniform(low=0.0, high=6.28, size=X_test.shape)
X_test_noisy = X_test + np.mod(3 * (phase_noise_coef * random_noise),6.28)
############################################################
num_noisy_sets = 5
random_noise = np.random.uniform(low=0.0, high=6.28, size=X_train.shape)
X_train_concat = X_train + random_noise
Y_train_concat = Y_train
for Xtrain_index in range(num_noisy_sets):
    random_noise = np.random.uniform(low=0.0, high=6.28, size=X_train.shape)
    X_tr_noisy = X_train + random_noise
    X_train_concat = np.vstack((X_train_concat,X_tr_noisy))
    print('X_train_concat.shape',X_train_concat.shape)
    
    Y_train_concat = np.vstack((Y_train_concat,Y_train))
    print('Y_train_concat.shape',Y_train_concat.shape)
    




num_batches = int(len(Y_train_concat) / batch_size)

####################### batch=16 and num_epochs = 1000 ######################################
del model
K.clear_session()
num_epochs = 20
# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(num_units, return_sequences=True, input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32 
# how many parameters do we have in the above layer? 4 * [num_units*num_units(Waa) + num_units*data_dim(Wax) + num_units(bias)] = 4*[20*20+20*100+20] = 4*121*20 = 9680
#model.add(BatchNormalization(axis=-1,momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'))
model.add(LSTM(num_units, return_sequences=True))  #  how many parameters do we have in the above layer? 
#model.add(BatchNormalization(axis=-1,momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'))
model.add(LSTM(num_units))  # how many parameters do we have in the above layer? 
#model.add(BatchNormalization(axis=-1,momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'))
#model.add(Dense(20, activation='relu'))
model.add(Dense(2, activation='softmax')) # how many parameters do we have in the above layer? 2*num_units(weights) + 2(bias) = 42
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.summary()


history = model.fit(X_train_concat, Y_train_concat, batch_size=batch_size, epochs=num_epochs,validation_data=(X_test_noisy, Y_test))

#model.save('.h5') # save the model

print(history.history.keys())
#  "Accuracy"
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy for batch 16 after 1000 epochs')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss for batch 16 after 1000 epochs')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
