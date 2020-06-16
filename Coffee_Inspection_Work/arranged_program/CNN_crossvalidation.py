import torch
from torch.autograd import Variable 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os 
from PIL import Image 
import numpy as np
import pandas as pd
from sklearn import datasets, model_selection
from sklearn.model_selection import train_test_split
import glob
import pandas as pd
from time import sleep
import pymongo
import gridfs 
import io
### crossValidation 관련 추가한 부분 
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer 
from sklearn.model_selection import KFold 

# 테스트 한번 할게용
# 테스트 두번 할게용
# 확인해주세요



#================================================================================================
# <CUDA 설정>
# 만약 GPU를 사용 가능하다면 device 값이 cuda가 되고, 아니라면 cpu가 됩니다.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device : " + device)
print("="*50)
# 랜덤 시드 고정
# torch.manual_seed(777)
# GPU 사용 가능일 경우 랜덤 시드 고정
# if device == 'cuda':
    # torch.cuda.manual_seed_all(777)
#================================================================================================


#================================================================================================
# <데이터 로드 (로컬)>


# # X
# normal_name_list = glob.glob('./Final_data/normal/*.png')
# broken_name_list = glob.glob('./Final_data/broken/*.png')
# black_name_list = glob.glob('./Final_data/black/*.png')

# img_filename_list = np.array(normal_name_list + broken_name_list + black_name_list)
# # img_filename = img_filename_list.reshape((img_filename_list.shape[0]),1)

# # Y 
# # 0 : normal
# # 1 : broken
# # 2 : black
# normal_label = [ 0 for label in range(len(normal_name_list))]
# broken_label = [ 1 for label in range(len(broken_name_list))]
# black_label =  [ 2 for label in range(len(black_name_list))]

# label_list = np.array(normal_label + broken_label + black_label)

# # print(img_filename.shape, label.shape) # (63888, 1) (63888, 1)

# CNT_DATA = len(label_list)
# img_list = list()
# for i in range(CNT_DATA):
#     img = np.array(Image.open(img_filename_list[i]).convert('L'))
#     img_list.append(img)
#     # if i == 100: break


############################################################
# <데이터 로드 및 준비 (데이터베이스))>
conn = pymongo.MongoClient('127.0.0.1', 27017)
#categories = ['normal', 'broken']
categories = ['normal', 'broken', 'black']

test = "test"
img_list = list()
label_list = list()
for idx, category in enumerate(categories):
    db = conn.get_database(category+'_rb')
    fs = gridfs.GridFS(db, collection = category)
    tmp = list()
    for f in fs.find():
        image = np.array(Image.open(io.BytesIO(f.read())).convert("L"))
        tmp.append(image)
    
    label = [ idx for label in range(len(list(tmp)))]
    label_list = label_list + label        
    img_list = img_list + tmp

label_list = np.array(label_list)
img_list = np.array(img_list)

# 여기서 데이터 구조 맞춰야 한다

print(label_list.shape, img_list.shape)
print("============")

X_data = torch.Tensor(np.array(img_list)).unsqueeze(1) # 1 channel
# X_data = torch.transpose(torch.Tensor(np.array(img_list)), 2,3) # 3 channel
# X_data = torch.transpose(X_data, 1,2) # 3 channel
print(X_data.shape)

y_data = torch.from_numpy(label_list).long()

#X_train, X_test, y_train, y_test = train_test_split(X_data,y_data, test_size = 0.25, random_state = 55)
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    




#================================================================================================




#================================================================================================
# 교차 검증시 다른 데이터 이용하기 위해 우선 주석처리 
# 학습할 때마다 dataset 재정의 
# dataset = TensorDataset(X_train, y_train)
# type(dataset)
#================================================================================================


#================================================================================================
# <학습 환경 설정>
learning_rate = 0.001
#learning_rate = 0.01
training_epochs = 500
#batch_size = int(56128 / training_epochs)
batch_size = 300

# data_loader = torch.utils.data.DataLoader(dataset=dataset,
#                                           batch_size=batch_size,
#                                           shuffle=True,
#                                           drop_last=True)
#================================================================================================


#================================================================================================
# <CNN 모델>
class CNN(torch.nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()
        # 첫번째층
        # ImgIn shape=(?, 64, 64, 1)
        #    Conv     -> (?, 64, 64, 32)
        #    Pool     -> (?, 32, 32, 32)
        ############
        # 채널 맞춰서 바꿔주세요
        ############
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1 ),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # 두번째층
        # ImgIn shape=(?, 32, 32, 32)
        #    Conv      ->(?, 32, 32, 64)
        #    Pool      ->(?, 16, 16, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # 세번째층
        # ImgIn shape=(?, 16, 16, 64)
        #    Conv      ->(?, 16, 16, 128)
        #    Pool      ->(?, 8, 8, 128)
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # 전결합층 8x8x128 inputs -> 3 outputs
        self.fc = torch.nn.Linear(8 * 8 * 128, 3, bias=True)

        # 전결합층 한정으로 가중치 초기화
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)   # 전결합층을 위해서 Flatten
        out = self.fc(out)
        return out
#================================================================================================

#================================================================================================
# <학습>

# CNN 모델 정의
model = CNN().to(device)
# 비용 함수와 옵티마이저를 정의합니다.
criterion = torch.nn.CrossEntropyLoss().to(device)    # 비용 함수에 소프트맥스 함수 포함되어져 있음.

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# 교차검증을 위해 우선 주석 처리 
# 총 배치의 수를 출력해보겠습니다.
# total_batch = len(data_loader)
# print('총 배치의 수 : {}'.format(total_batch))
# 총 배치의 수 : 600
# 총 배치의 수는 600입니다. 그런데 배치 크기를 100으로 했으므로 결국 훈련 데이터는 총 60,000개란 의미입니다. 

# 이제 모델을 훈련시켜보겠습니다.
model.train()

# 교차검증관련 
# 3번 검증 
kf = KFold(n_splits=3, random_state=None, shuffle=True)

for epoch in range(training_epochs):
    
    # 교차검증을 위해 추가한 부분 
    for train_index, test_index in kf.split(X_data):
        X_train = X_data[train_index]
        y_train = y_data[train_index]
        X_test = X_data[test_index]
        y_test = y_data[test_index]

        dataset = TensorDataset(X_train, y_train)
        testset = TensorDataset(X_test, y_test)
        
        # # <학습 및 테스트 환경 설정>
        # learning_rate = 0.001
        # training_epochs = 1
        # batch_size = int(56128 / training_epochs)
        # batch_size = 300

        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)
        
        testloader = torch.utils.data.DataLoader(dataset=testset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)



        

        total_batch = len(data_loader)
        print('총 배치의 수 : {}'.format(total_batch))

        avg_cost = 0.0

        for X, Y in data_loader: # 미니 배치 단위로 꺼내온다. X는 미니 배치, Y는 레이블.
            # image is already size of (64x64), no reshape
            # label is not one-hot encoded
            X = X.to(device)
            Y = Y.to(device)

            optimizer.zero_grad()
            hypothesis = model(X)

            cost = criterion(hypothesis, Y)
            cost.backward()
            optimizer.step()

            avg_cost += cost / total_batch
        
        
        

        # 테스트 데이터로 모델 테스트 진행 
        """
        https://wingnim.tistory.com/36
        """

        model.eval()
        test_loss = 0 
        correct = 0

        for data, target in testloader: 
            data = data.to(device)
            target = target.to(device) 

            output = model(data)
            # print(output)
            # sleep(1212)

            # sum up batch loss 
            test_loss += criterion(output, target).data

            # get the index of the max log-probability 
            pred = output.data.max(1, keepdim = True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            test_loss /= len(testloader.dataset)/batch_size

        print('\nTest set : Average loss : {: .4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(testloader.dataset), 100. * correct / len(testloader.dataset)))
    
    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))




#############################################################
#############################################################
#############################################################
################################################## 모델 세이브
# crossValidation model 별도 저장 위해, 기존 경로에서 파일명 수정 
model_path = "./Models/test_model_crossValidation.pth"
torch.save(model.state_dict(), model_path)




# batch_size = CNT_DATA /100 (한 에포크에 100번 돌도록)
# (?, 64, 64, 1) > (?, 32, 32, 32) > (?, 16, 16, 64) > (?, 8, 8, 128) > 3
# first epoch
    # Test set : Average loss :  0.3298, Accuracy: 14553/15972 (91%)

# fc망
# 8 > 512(추가) -> 8
# 망을 하나 늘렸는데 더 안좋아짐.
# second epoch
    # Test set : Average loss :  0.4893, Accuracy: 14081/15972 (88%)

# batch size = 100(한 에포크 : 479번 반복)
# third epoch
    # Test set : Average loss :  0.8559, Accuracy: 14097/15972 (88%)

# batch size = 30
# fourth epoch

# 학습속도가 빠르고 / 과적합의 우려가 있음
# 시도해볼만한 것
# learning_rate 조정
# Epoch 특정 지점에서 stop시키는 것
# batch_size 조정
# black 빼서 시도

# 해야하는 것
# 모델 세이브

