# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 14:39:14 2021

@author: CNL-B3
"""
def solution(s):
    answer = 0
    engNumber = ["zero", "one", "two", "three", "four", "five", "six","seven", "eight", "nine"]
    stringofAnswer = ''
    for i in range(0,len(s)):
        if(s[i].isdigit()):
            stringofAnswer += s[i]
        else:
            stringOfNumberindex = -1
            for stringOfNumber in engNumber:
                stringOfNumberindex += 1
                tempData = s[i:i + len(stringOfNumber) + 1]
                index = tempData.find(stringOfNumber)
                if index == 0:
                    stringofAnswer += str(stringOfNumberindex)
                    break
                
                    
    answer = int(stringofAnswer)         
    return answer

print(solution("one4seveneight"))
