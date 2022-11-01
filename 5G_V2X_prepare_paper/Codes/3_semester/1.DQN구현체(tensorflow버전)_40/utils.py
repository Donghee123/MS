import time
import numpy as np
import _pickle as cPickle
import csv
import os
import pandas as pd
    
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