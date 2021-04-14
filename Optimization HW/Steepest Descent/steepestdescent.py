# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 00:58:37 2021

@author: Handonghee
"""

import numpy as np

#Qx - b
def firstOrder(q, b, x):
    answer = (q @ x) - b
    return answer
    
def secondOrder(q):
    return q

#================steepestdescent========================
def getRate(q, b, x):
    child = np.transpose(firstOrder(q, b, x)) @ firstOrder(q, b, x)
    parent = np.transpose(firstOrder(q, b, x)) @ q @ firstOrder(q, b, x)
    return child / parent   

def getNextvector(q, b, x):
    a = getRate(q, b, x)    
    return x - a * firstOrder(q, b, x)

def steepest_method(q,b,x): 
      
    nextX = getNextvector(q, b, x)
    preX = nextX.copy()

    for i in range(100):       
        nextX = getNextvector(q, b,preX)   
        
        
        if (abs(nextX-preX) < 0.000001).all():
            break
        
        preX = nextX.copy()    
        print('iteration count : ' + str(i))
        print(preX)
        
    print('optimal X is')    
    print(nextX)
#=======================================================
    
def homework3(method_name, demension, q, b, x):    
    if method_name == "steepest":
        steepest_method(q,b,x)
    

def newton_method(q,b):
    return 0

    
inputQ = np.array([[3,0,1],[0,4,2],[1,2,3]])
inputB = np.array([[3],[0],[1]])
inputX = np.array([[0],[0],[1]])

homework3('steepest', 2, inputQ, inputB, inputX)