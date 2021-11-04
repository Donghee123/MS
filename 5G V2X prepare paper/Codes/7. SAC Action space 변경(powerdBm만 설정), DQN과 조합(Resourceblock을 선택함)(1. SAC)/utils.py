import math
import torch
import time
import numpy as np
import _pickle as cPickle
import csv
import os
import pandas as pd

def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
        
#File 유틸 함수들    
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
     
def MakeCSVFile(folderpath, aryOfDatas, useCaseFilename, memoryIndex):
    df = pd.DataFrame(aryOfDatas)
    
    fileName = folderpath + '/memory_' + useCaseFilename + '_' + str(memoryIndex) + '.csv'
    
    df.to_csv(fileName, index=False)

def save_pkl(obj, path):
  with open(path, 'wb') as f:
    cPickle.dump(obj, f)
    print("  [*] save %s" % path)
    
def load_pkl(path):
  with open(path, 'rb') as f:
    obj = cPickle.load(f)
    print("  [*] load %s" % path)
    return obj        
