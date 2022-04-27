from tkinter.messagebox import NO
import numpy as np   
import torch
from network import Network
from dataloader import *
import torch.optim as optim
import os
from sklearn.metrics import confusion_matrix

def train(model, nbatch_size, train_samples, optimizer, criterion, epoch, nlog_interval):

    model.train()
    train_loader = torch.utils.data.DataLoader(train_samples, batch_size=nbatch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nBatchCount = 0
    for data, target in train_loader:
        nBatchCount += 1
        data = data.float().to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if nBatchCount % nlog_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, nBatchCount * len(data), len(train_loader.dataset), 100. * nBatchCount / len(train_loader), loss.item()))

def test(model, dev_samples, nBatchSize):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    true_y_list = []
    pred_y_list = []

    with torch.no_grad():
        test_loader = torch.utils.data.DataLoader(dev_samples, batch_size=nBatchSize, shuffle=True)

        for data, true_y in test_loader:
            data = data.float().to(device)
            true_y = true_y.to(device)           
                
            output = model(data)
            pred_y = torch.argmax(output, axis=1)

            pred_y_list.extend(pred_y.tolist())
            true_y_list.extend(true_y.tolist())

    train_accuracy =  accuracy_score(true_y_list, pred_y_list)
    return train_accuracy

def getconfusionmetrix(model, dev_samples):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    true_y_list = []
    pred_y_list = []

    with torch.no_grad():
        test_loader = torch.utils.data.DataLoader(dev_samples, batch_size=nBatchSize, shuffle=True)

        for data, true_y in test_loader:
            data = data.float().to(device)
            true_y = true_y.to(device)           
                
            output = model(data)
            pred_y = torch.argmax(output, axis=1)

            pred_y_list.extend(pred_y.tolist())
            true_y_list.extend(true_y.tolist())

    return confusion_matrix(true_y_list, pred_y_list, labels=[0, 1, 2, 3])


def accuracy_score(label_y, predict_y):
    
    correct_count = 0
    for index, value in enumerate(label_y):
        if predict_y[index] == value:
            correct_count += 1
    
    return correct_count / len(label_y)


fLearningRate = 0.001
nEpoch = 30
nBatchSize = 128
nlog_interval = 5

nInputSize = 132
nOutputSize = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Network(nInputSize, nOutputSize).to(device)
optimizer = optim.Adam(model.parameters(), lr=fLearningRate)
       
criterion = torch.nn.CrossEntropyLoss()

# If you want to use full Dataset, please pass None to csvpath
strDataFolderPath = os.path.join('sample_image_folder', 'skeleton_npy')
skeleton_samples = SkeletonDataset(strDataFolderPath) 

train_size = int(0.8 * len(skeleton_samples))
test_size = len(skeleton_samples) - train_size

train_set, val_set = torch.utils.data.random_split(skeleton_samples, [train_size, test_size])

fmaxAcc = 0

str_BestModelpath = ''
for epoch in range(1, nEpoch):
    train(model, nBatchSize, train_set, optimizer, criterion, epoch, nlog_interval)
    test_acc = test(model, val_set, nBatchSize)
    print('Dev accuracy ', test_acc)
    if fmaxAcc < test_acc:
        fMaxAcc = test_acc
        str_BestModelpath = f'./models/model_{test_acc}.pkl'
        torch.save(model.state_dict(),str_BestModelpath)
        
testmodel = Network(nInputSize, nOutputSize).to(device)
testmodel.load_state_dict(torch.load(str_BestModelpath))
print(getconfusionmetrix(testmodel, val_set))
    
    
    
