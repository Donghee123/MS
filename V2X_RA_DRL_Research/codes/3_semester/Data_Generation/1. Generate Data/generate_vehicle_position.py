import random
import numpy as np
import os
import sys
import pandas as pd
import csv

from Environment import *

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
    
# ################## SETTINGS ######################
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

IS_TRAIN = 1
IS_TEST = 1-IS_TRAIN

generateSecond = 20
n_veh = 60
n_neighbor = 1

listOfVehiclePosition = []

env = Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height, n_veh)  # V2X 환경 생성

env.add_new_vehicles_by_number(int(n_veh/4))

for i in range((generateSecond * 10)):
    
    env.renew_positions()
    positions = [c.position for c in env.vehicles]
    
    print(positions)
    tempvalue = []
    for j in range(0, n_veh):
       tempvalue.append((positions[j][0], positions[j][1]))
    
    listOfVehiclePosition.append(tempvalue)
    

print('and...')
for j in range(len(listOfVehiclePosition)):
    print(listOfVehiclePosition[j])

listOfVehiclePosition = np.array(listOfVehiclePosition)
dataPath = 'position/'

vehicleheader = []

for i in range(n_veh):
    vehicleheader.append('vehicle' + str(i + 1))
  
MakeCSVFile(dataPath, 'vehiclePosition.csv', vehicleheader, listOfVehiclePosition)
print('position save 완료')
