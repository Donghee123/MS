import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
import cv2

from network import Network
from dataloader import *

if __name__ == '__main__':   
    nInputSize = 132
    nOutputSize = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Network(nInputSize, nOutputSize).to(device)
    PATH = "testmodel/model.pkl"
    model.load_state_dict(torch.load(PATH))

    model.eval()

    strDataFolderPath = os.path.join('for_presentation', 'skeleton_npy')
    strImageFolderPath = os.path.join('for_presentation', 'testvideo')
    strsaveImageFolderPath = os.path.join('for_presentation', 'test_predictrion_result')

    skeleton_samples = SkeletonTestDataset(strDataFolderPath) 
    skeleton_dataloader = DataLoader(skeleton_samples, batch_size=1, shuffle=False)
    softMax = nn.Softmax(dim=1)

    dictOfLabes = {0 : 'good', 1 : 'bad', 2: 'bad', 3: 'bad'}


    Green = (22,222,22)
    Red = (22, 22, 222)
    for filename, skeleton in skeleton_dataloader:
        textColor = Green
        skeleton = skeleton.to(device).float()               
        output = model(skeleton)
        output = softMax(output)

        nIndex = torch.argmax(output).to('cpu').detach()
        predictIndex = int(torch.argmax(output).to('cpu').detach())

        if nIndex == 3:
            prob = output[0][nIndex]
            if prob <= 0.95:
                predictIndex = 0

        print(dictOfLabes[predictIndex])
        str_predict_pose = dictOfLabes[predictIndex]
        image = cv2.imread(f'{strImageFolderPath}\\{filename[0]}.jpg')
        font = cv2.FONT_HERSHEY_DUPLEX # 텍스트의 폰트를 지정. cv2.putText(img, "WARNNING!! This is DDOS Virus", (20, 90), font, 2,(0,0,155), 2, cv2.LINE_AA)

        if predictIndex > 0:
            textColor = Red
        

        cv2.putText(image, f"{str_predict_pose}", (20, 120), font, 3,textColor, 2, cv2.LINE_AA)
        if predictIndex == 3:
            cv2.putText(image, 'forward head posture', (20, 200), font, 3,textColor, 2, cv2.LINE_AA)
        if predictIndex == 1:
            cv2.putText(image, 'left leaning posture', (20, 200), font, 3,textColor, 2, cv2.LINE_AA)
        if predictIndex == 2:
            cv2.putText(image, 'right leaning posture', (20, 200), font, 3,textColor, 2, cv2.LINE_AA)
            
        
        savePath = os.path.join(strsaveImageFolderPath, f'{filename[0]}.jpg')
        cv2.imwrite(savePath, image)


