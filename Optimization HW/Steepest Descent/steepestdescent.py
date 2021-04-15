# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 00:58:37 2021

@author: Handonghee
"""

import numpy as np

#Firstorder Qx - b
def firstderivative(q, b, x):
    answer = (q @ x) - b
    return answer
    
def secondderivative(q):
    return q

def getinversmatix(q):
    return np.linalg.inv(q)

#================steepestdescent========================
def steepestdescent_getRate(q, b, x):
    child = np.transpose(firstderivative(q, b, x)) @ firstderivative(q, b, x)
    parent = np.transpose(firstderivative(q, b, x)) @ q @ firstderivative(q, b, x)
    return child / parent   

def steepestdescent_getNextvector(q, b, x):
    a = steepestdescent_getRate(q, b, x)    
    return x - a * firstderivative(q, b, x)

def steepest_method(q,b,x): 
      
    nextX = steepestdescent_getNextvector(q, b, x)
    preX = nextX.copy()
    print('steepset : iteration count : 0')
    print(preX)
    
    for i in range(100):       
        nextX = steepestdescent_getNextvector(q, b,preX)   
        
        
        if (abs(nextX-preX) < 0.000001).all():
            break
        
        preX = nextX.copy()    
        print('steepset : iteration count : ' + str(i + 1))
        print(preX)
        
    print('steepset : iteration count : ' + str(i + 1) + ',optimal X is')    
    print(nextX)
#=======================================================
#================newton method==========================  
def newtonmethd_getRate(q, b, x):    
    return getinversmatix(secondderivative(q)) 

def newtonmethd_getNextvector(q,b,x):
    return x - (newtonmethd_getRate(q,b,x) @ firstderivative(q, b, x))

def newton_method(q,b,x): 
      
    nextX = newtonmethd_getNextvector(q, b, x)
    preX = nextX.copy()
    print('newton : iteration count : 0')
    print(preX)
    
    for i in range(100):       
        nextX = newtonmethd_getNextvector(q, b, preX)   
               
        if (abs(nextX-preX) < 0.000001).all():
            break
        
        preX = nextX.copy()    
        print('newton : iteration count : ' + str(i + 1))
        print(preX)
        
    print('newton : iteration count : ' + str(i + 1) + ', optimal X is')    
    print(nextX)
#=======================================================  
#================conjugate direction method=============
def conjugate_getRate(q, b, x, d):
    child = np.transpose(firstderivative(q, b, x)) @ d
    parent = np.transpose(d) @ q @ d
    return -child/parent

def conjugate_getBeta(q, b, x, d):
    child = np.transpose(firstderivative(q, b, x)) @ q @ d
    parent = np.transpose(d) @ q @ d
    return child/parent

def conjugate_method(q,b,x):
    #처음 direction, d0 설정
    direction = -firstderivative(q, b, x)
    #처음 alpha 구하기
    alpha = conjugate_getRate(q,b,x,direction)
    #다음 iteration X 구하기
    nextX = x + (alpha * direction)
    #다음 iteration beta 구하기
    beta = conjugate_getBeta(q,b,nextX, direction)
    #처음 direction, d1 구하기
    direction = (beta * direction) - firstderivative(q, b, nextX)
    
    print('conjugate : iteration count : 0')
    print(nextX)
    
    preX = nextX.copy()
    
    for i in range(500):
        #alpha 계산 
        alpha = conjugate_getRate(q,b,nextX,direction)
        #다음 iteration X 구하기
        nextX = nextX + (alpha * direction)
        #다음 iteration beta 구하기
        beta = conjugate_getBeta(q,b,nextX, direction)
        #처음 direction, d1 구하기
        direction = (beta * direction) - firstderivative(q, b, nextX)
        
        print('conjugate : iteration count : ' + str(i + 1))
        print(nextX)
        
        if (abs(nextX-preX) < 0.000001).all():
            break
        
        preX = nextX.copy()
        
   
    print('conjugate : iteration count : ' + str(i + 1) + ', optimal X is')    
    print(nextX)
    
#=======================================================    
def homework3(method_name, demension, q, b, x):    
    if method_name is "steepest":
        steepest_method(q,b,x)
    elif method_name is "newton":
        newton_method(q,b,x)
    elif method_name is "conjugate":
        conjugate_method(q,b,x)

    
inputQ = np.array([[3,0,1],[0,4,2],[1,2,3]])
inputB = np.array([[3],[0],[1]])
inputX = np.array([[-10],[10],[1]])
#inputQ = np.array([[5,2],[2,1]])
#inputB = np.array([[3],[1]])
#inputX = np.array([[0],[0]])

homework3('conjugate', 2, inputQ, inputB, inputX)