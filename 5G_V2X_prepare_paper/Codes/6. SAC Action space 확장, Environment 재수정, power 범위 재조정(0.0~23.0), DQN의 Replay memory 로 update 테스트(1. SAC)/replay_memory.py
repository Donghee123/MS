import random
import numpy as np
import pandas as pd
from utils import *

class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
     
    def push_preconcatenate(self, concatenateData):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
            
        
        concatenateData.append(1) #to input done
        #state(82), action(2), reward(1), nextstate(82), done(1)
        self.buffer[self.position] = (np.array(concatenateData[0:82], dtype=float), np.array(concatenateData[82:84], dtype=float), np.array(concatenateData[84:85],dtype=float),np.array( concatenateData[85:167],dtype=float), np.array(concatenateData[167:168], dtype=float))
        self.position = (self.position + 1) % self.capacity
        
    def load(self, Filepath):
        array = []
        
        with open(Filepath) as file_name:
            file_read = csv.reader(file_name)
            array = list(file_read)
        
        for index in range(1, len(array)):
            self.push_preconcatenate(array[index])
        
    def save(self, useCaseFilename):        
        folderpath = './model/train'        
        createFolder(folderpath)       
        datas = []
        
        for index in range(len(self.buffer)):                 
            datas.append(self.buffer[index])
                
        MakeCSVFile(folderpath, datas, useCaseFilename, self.saveCount)
        self.saveCount += 1
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
