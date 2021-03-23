# -*- coding: utf-8 -*-
"""
DNN based basestation scheduling model
excute train
excute test
save model
분류기(Classifier) 학습하기
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils 
import pandas as pd
import MakeMRSchdulingModel_8UETo16Target as MRSchedulingUtil

"""
Start Pandas API를 이용한 엑셀 데이터 취득 구간
"""
#100000개 데이터 수집
trndataset = pd.read_csv('./dataset/train/batch0/dataset.csv')
"""
End Pandas API를 이용한 엑셀 데이터 취득 구간
"""


"""
Start 훈련 및 테스트용 데이터 분할 구간
"""
seed = 1

#epochin
epoch = 50
#해당 각 데이터 범주 정의
X_features = ["UE0", "UE1", "UE2", "UE3","UE4", "UE5", "UE6", "UE7"]
y_features = ["1_SelUE0", "1_SelUE1", "1_SelUE2", "1_SelUE3", "1_SelUE4", "1_SelUE5", "1_SelUE6", "1_SelUE7", 
                 "2_SelUE0", "2_SelUE1", "2_SelUE2", "2_SelUE3", "2_SelUE4", "2_SelUE5", "2_SelUE6", "2_SelUE7"]

#batch size = 1
batch_size = 1

#범주 별로 데이터 취득 X : 입력, Y : 출력
trn_X_pd, trn_y_pd = trndataset[X_features], trndataset[y_features ]

#pandas.core.frame.DataFrame -> numpy -> torch 데이터형으로 변경
trn_X = torch.from_numpy(trn_X_pd.astype(float).to_numpy())
trn_y = torch.from_numpy(trn_y_pd.astype(float).to_numpy())

#torch DataSet(input, output) 세트로 변경
trn = data_utils.TensorDataset(trn_X, trn_y)


#데이터셋 중 훈련 : 70%, 검증 : 10% 사용
trainsetSize = int(70 * len(trn) / 100)
valisetSize = int(10 * len(trn) / 100)
testsetSize = len(trn) - (trainsetSize + valisetSize) 

trn_set, val_set, test_set = torch.utils.data.random_split(trn, [trainsetSize, valisetSize, testsetSize])

#훈련용 데이터 로더
trn_loader = data_utils.DataLoader(trn_set, batch_size=batch_size, shuffle=True)
#검증용 데이터 로더
val_loader = data_utils.DataLoader(val_set, batch_size=batch_size, shuffle=True)
#테스트용 데이터 로더
test_loader = data_utils.DataLoader(test_set, batch_size=batch_size, shuffle=True)

#범주별 data_loader 핸들링
data_loaders = {"train": trn_loader, "val": val_loader, "test":test_loader}
"""
End 훈련 및 테스트용 데이터 분할 구간
"""



"""
start module 인스턴스 Loas 후 학습 및 테스트 구간
"""
torch.manual_seed(seed)
strModelPath = './model/model.pt'
model = torch.load(strModelPath)
#MES Loss
#criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
#최적화 함수 SGD, 학습률 0.001, momentum 0.5
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)

aryofLoss = []
aryofModelInfo = []
aryofValidationLoss = []
log_interval = 1
           
for i in range(0,epoch):
    #훈련
    MRSchedulingUtil.train(i, data_loaders['train'], data_loaders['val'],2000)
    #평가
    MRSchedulingUtil.test(log_interval, model, data_loaders['test'])
"""
end module 인스턴스 Loas 후 학습 및 테스트 구간
"""

"""
start save module구간
"""
modelPath = "./model"
MRSchedulingUtil.createFolder(modelPath)   
MRSchedulingUtil.MakeCSVFile(modelPath, "ModelLossInfo.csv", ["Loss"], aryofLoss)
MRSchedulingUtil.MakeCSVFile(modelPath, "ModelValidationLossInfo.csv", ["Validation Loss"], aryofValidationLoss)
MRSchedulingUtil.MakeCSVFile(modelPath, "ModelSpec.csv",["Average Loss","Correct","Accracy"],aryofModelInfo)
torch.save(model, modelPath + '/model.pt')    
"""
end save module구간
"""
