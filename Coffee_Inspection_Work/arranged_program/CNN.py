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


#================================================================================================
# <CUDA 설정>
# 만약 GPU를 사용 가능하다면 device 값이 cuda가 되고, 아니라면 cpu가 됩니다.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 랜덤 시드 고정
torch.manual_seed(777)
# GPU 사용 가능일 경우 랜덤 시드 고정
if device == 'cuda':
    torch.cuda.manual_seed_all(777)
#================================================================================================


#================================================================================================
# <데이터 로드 및 준비 (로컬)>


# X
# normal_name_list = glob.glob('./Final_data/normal/*.png')
# broken_name_list = glob.glob('./Final_data/broken/*.png')
# black_name_list = glob.glob('./Final_data/black/*.png')

# img_filename_list = np.array(normal_name_list + broken_name_list + black_name_list)
# # img_filename = img_filename_list.reshape((img_filename_list.shape[0]),1)

# Y 
# 1 : normal
# 2 : broken
# 3 : black
# normal_label = [ 0 for label in range(len(normal_name_list))]
# broken_label = [ 1 for label in range(len(broken_name_list))]
# black_label =  [ 2 for label in range(len(black_name_list))]

# label_list = np.array(normal_label + broken_label + black_label)

# CNT_DATA = len(label_list)
# img_list = list()
# for i in range(CNT_DATA):
#     img = np.array(Image.open(img_filename_list[i]).convert('L'))
#     img_list.append(img)



# <데이터 로드 및 준비 (데이터베이스))>
conn = pymongo.MongoClient('127.0.0.1', 27017)
categories = ['normal', 'broken', 'black']

test = "test"
img_list = list()
label_list = list()
for idx, category in enumerate(categories):
    db = conn.get_database(category)
    fs = gridfs.GridFS(db, collection = category)
    tmp = list()
    for f in fs.find():
        image = np.array(Image.open(io.BytesIO(f.read())))
        tmp.append(image)
    label = [ 0 for label in range(len(list(tmp)))]
    label_list = label_list + label        
    img_list = img_list + tmp

label_list = torch.Tnp.array(label_list)
img_list = np.array(img_list)

# 여기서 데이터 구조 맞춰야 한다

print(label_list.shape, img_list.shape)
print("============")
sleep(10)


X_data = torch.Tensor(np.array(img_list)).unsqueeze(1)
y_data = torch.from_numpy(label_list).long()


X_train, X_test, y_train, y_test = train_test_split(X_data,y_data, test_size = 0.25, random_state = 55)
dataset = TensorDataset(X_train, y_train)
#================================================================================================


#================================================================================================
# <학습 환경 설정>
learning_rate = 0.001
training_epochs = 100
batch_size = int(63888 / training_epochs)

data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)
#================================================================================================


#================================================================================================
# <CNN 모델>
class CNN(torch.nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()
        # 첫번째층
        # ImgIn shape=(?, 1, 64, 64)
        #    Conv     -> (?, 32, 64, 64)
        #    Pool     -> (?, 32, 32, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1 ),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # 두번째층
        # ImgIn shape=(?, 32, 32, 32)
        #    Conv      ->(?, 64, 32, 32)
        #    Pool      ->(?, 64, 16, 16)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # 세번째층
        # ImgIn shape=(?, 64, 16, 16)
        #    Conv      ->(?, 128, 16, 16)
        #    Pool      ->(?, 128, 8, 8)
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
# 총 배치의 수를 출력해보겠습니다.
total_batch = len(data_loader)
print('총 배치의 수 : {}'.format(total_batch))

# 훈련
model.train()

for epoch in range(training_epochs):
    
    avg_cost = 0

    for X, Y in data_loader: # 미니 배치 단위로 꺼내온다. X는 미니 배치, Y는 레이블.
        # image is already size of (64x64), no reshape
        # label is not one-hot encoded
        X = X.to(device)
        Y = Y.to(device)
        # print(type(X))
        # print(type(Y))
        # print(X.shape)
        # print(Y.shape)
        # print(X)
        # print(Y)
        # sleep(12)

        optimizer.zero_grad()
        hypothesis = model(X)

        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))


# test dataset 만들기
testset = TensorDataset(X_test, y_test)
type(testset)

batch_size = 100

testloader = torch.utils.data.DataLoader(dataset=testset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          drop_last=True)

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

    # sum up batch loss 
    test_loss += criterion(output, target).data

    # get the index of the max log-probability 
    pred = output.data.max(1, keepdim = True)[1]
    correct += pred.eq(target.data.view_as(pred)).cpu().sum()

test_loss /= len(testloader.dataset)/batch_size

print('\nTest set : Average loss : {: .4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(testloader.dataset), 100. * correct / len(testloader.dataset)))
