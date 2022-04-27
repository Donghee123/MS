import os
import pathlib
import numpy as np
from torch.utils.data import Dataset


class SkeletonTestDataset(Dataset):

    def __init__(self, strFolderPath : str):
        
        strDataFolderlist = os.listdir(strFolderPath)
        
        self.X = []
        
        for strLabelFolderPath in strDataFolderlist:
            strOneLabelDataPath = os.path.join(strFolderPath, strLabelFolderPath)            
            self.X.append(strOneLabelDataPath)                

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        x = np.load(self.X[idx], allow_pickle=True).flatten()     
        filename = pathlib.PurePath(self.X[idx])  
        filename = pathlib.Path(filename).stem
        return filename, x

class SkeletonDataset(Dataset):

    def __init__(self, strFolderPath : str):
        self.dictOfLabes = {'good' : 0, 'left' : 1, 'right' : 2, 'turtleneck' : 3}
        strDataFolderlist = os.listdir(strFolderPath)
        
        self.X = []
        self.Y = []

        for strLabelFolderPath in strDataFolderlist:
            strOneLabelDataPath = os.path.join(strFolderPath, strLabelFolderPath)
            listOfOneLabelDataPath = os.listdir(strOneLabelDataPath)
            for strNPDataPath in listOfOneLabelDataPath:
                self.X.append(os.path.join(strOneLabelDataPath,strNPDataPath))
                self.Y.append(self.dictOfLabes[strLabelFolderPath])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = np.load(self.X[idx], allow_pickle=True).flatten()
        y = self.Y[idx]
        return x, y