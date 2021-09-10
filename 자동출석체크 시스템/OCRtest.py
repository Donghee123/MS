# -*- coding: utf-8 -*-
''' 
reference link : https://junyoung-jamong.github.io/computer/vision,/ocr/2019/01/30/Python%EC%97%90%EC%84%9C-Tesseract%EB%A5%BC-%EC%9D%B4%EC%9A%A9%ED%95%B4-OCR-%EC%88%98%ED%96%89%ED%95%98%EA%B8%B0.html 
Have to download OCR program firstly. Link : https://github.com/UB-Mannheim/tesseract/wiki 
environment : Window 10, python 3.6 
tesseract ver : v5.0.0.20190526.exe 

- 이미지 글자 추출 테스트 
cmd > tesseract path\name.png stdout 
- 패키지 설치
 pip install pillow
 pip install pytesseract
 pip install opencv-python 
 ''' 
import cv2 
import os 
try: 
     from PIL import Image
except ImportError: 
     import Image 
     
import pytesseract
import numpy as np
import pandas as pd

 # 32비트인 경우 => r'C:\Program Files (x86)\Tesseract-OCR\tesseract' 
 

def getFileList(folderPath):
    fileList = os.listdir(folderPath)
    answerFileList = []
    for fileName in fileList:
        answerFileList.append(folderPath + '\\' + fileName)
        
    return answerFileList

def GetNameList(loadImagePath, savePath):   
    
    # 설치한 tesseract 프로그램 경로 (64비트)
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract' 

    image = cv2.imread(loadImagePath) 
    
    #bgr to gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    
    #이미지 크기 조절
    gray = cv2.resize(gray, dsize=(0, 0), fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
        
    #컨볼루션
    sharpening = np.array([[-1, -1, -1, -1, -1],
                         [-1, 2, 2, 2, -1],
                         [-1, 2, 9, 2, -1],
                         [-1, 2, 2, 2, -1],
                         [-1, -1, -1, -1, -1]]) / 9.0
    
    gray = cv2.filter2D(gray, -1, sharpening)
    
    
    
    # write the grayscale image to disk as a temporary file so we can 
    # 글자 프로세싱을 위해 Gray 이미지 임시파일 형태로 저장. 
    #filename = '{}.bmp'.format(os.getpid()) 
    filename = savePath
    cv2.imwrite(filename, gray) 
    
    # Simple image to string 
    text = pytesseract.image_to_string(Image.open(filename), lang='kor')
    
    
    #1차 필터링 : 줄 단위
    save = text.split('\n')
    
    #2차 필터링 : 글자수 6개 이상
    #3차 필터링 : 글자에 숫자가 7개이상
    
    firstfilter = []

    for name in save:
        if len(name) >= 6 and len(name) <= 15:
            digitCount = 0
            
            for checkDigit in name:
                if checkDigit.isdigit():
                    digitCount += 1
            
            if digitCount >= 7:
                firstfilter.append(name)
    
                      
    return firstfilter


                
        
def GetAllNameList(folderPath):
    
    fileList = getFileList(folderPath)
    AllNameList = []
    
    for filePath in fileList:       
        file_name, file_ext = os.path.splitext(filePath)
        savefilelist = list(file_name)
                           
        savefilelist.insert((len(savefilelist)), '_.bmp')
        saveAfterProcessImagePath = ''.join(savefilelist)
        temp = GetNameList(filePath, saveAfterProcessImagePath)
        
        for name in temp:
            AllNameList.append(name)
    
    #모든 이름 내림차순 정렬
    
    AllNameList.sort()
    
    return AllNameList

def UpdateCheckList(excelPath):
    df = pd.read_excel(excelPath)
   
    result = []
    
    for index in range(len(df['number'])):
        
        temp = []
        temp.append(str(df['number'][index]))
        temp.append(df['name'][index])
        temp.append(False)
        result.append(temp)
                    
    return result
  
def GetResult(excelPath, nameList, result):
    
    df = pd.read_excel(excelPath)   
    
    for index in range(len(df['number'])):
        
        for checkAttendance in nameList:
            if checkAttendance.find(str(df['number'][index])) >= 0 or checkAttendance.find(str(df['name'][index])) >= 0:
                result[index][2] |= True
                break
        else:
            result[index][2] |= False
    
    return result

#집 전용 경로

#exelPath = "C:\\Users\\Handonghee\\anaconda3\\envs\\attendanceCheck\\CheckFolder\\referencesheet\\CheckList.xlsx"
#imagePath = "C:\\Users\\Handonghee\\anaconda3\\envs\\attendanceCheck\\CheckFolder\\checkimages"

#노트북 전용 경로
#exelPath = "C:\\Users\\gkseh\\anaconda3\\envs\\attendancecheck\\CheckFolder\\referencesheet\\CheckList.xlsx"
#imagePath = 'C:\\Users\\gkseh\\anaconda3\\envs\\attendancecheck\\CheckFolder\\checkimages'

#랩실 전용 경로
exelPath = "C:\\Users\\CNL-B3\\anaconda3\\envs\\attendanceCheck\\CheckFolder\\referencesheet\\CheckList.xlsx"
imagePath = "C:\\Users\\CNL-B3\\anaconda3\\envs\\attendanceCheck\\CheckFolder\\checkimages"

result = UpdateCheckList(exelPath)

#총 4번 다시 체크 이미지 사이즈 조절 하면서 체크 많이 할수록 정확도가 올라감
for i in range(6):
    arr = GetAllNameList(imagePath)
    result = GetResult(exelPath, arr, result)
    
for re in result:
    if re[2] == False:
        print(re)