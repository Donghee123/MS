import time
import numpy as np
import _pickle as cPickle
def save_pkl(obj, path):
  with open(path, 'wb') as f:
    cPickle.dump(obj, f)
    print("  [*] save %s" % path)
def load_pkl(path):
  with open(path, 'rb') as f:
    obj = cPickle.load(f)
    print("  [*] load %s" % path)
    return obj

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
    