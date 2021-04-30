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

 # 설치한 tesseract 프로그램 경로 (64비트)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract' 
 # 32비트인 경우 => r'C:\Program Files (x86)\Tesseract-OCR\tesseract' 
 
 # 이미지 불러오기, Gray 프로세싱
path = "C:\\Users\\CNL-B3\\Desktop\\test.jpg"
image = cv2.imread(path) 
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

# write the grayscale image to disk as a temporary file so we can 
# 글자 프로세싱을 위해 Gray 이미지 임시파일 형태로 저장. 
filename = '{}.png'.format(os.getpid()) 
cv2.imwrite(filename, gray) 

# Simple image to string 
text = pytesseract.image_to_string(Image.open(filename), lang='kor')
print(text)
os.remove(filename)
#cv2.imshow('Image', image) 
#cv2.waitKey(0)
