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


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# img_path = "./data/normal_data/normal_sec_processed_data/20200518_0_normal_take8_final.jpg"
# img_path = "./data/broken_data/broken_sec_processed_data/20200519_5_broken_take4_final.jpg"

model_path = "./Models/test_model.pth"

model = CNN().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

normal_cnt = 0
broken_cnt =0
for i in range(1,100, 1):
    # img_path = "./Final_data/broken/broken1st_pcd_"+ str(i) +".png"
    # img_path = "./data/broken_data/broken_sec_processed_data/20200519_" + str(i) + "_broken_take4_final.jpg"
    # img_path = "./data/normal_data/normal_sec_processed_data/20200518_" + str(i) + "_normal_take6_final.jpg"
    # img_path = "./Final_data/normal/normal1st_pcd_"+ str(i) +".png"
    # img_path = 

    img = torch.Tensor(np.array(Image.open(img_path).convert("L")))
    img = torch.unsqueeze(img, 0)
    img = torch.unsqueeze(img, 0)

    img = img.to(device)

    testset = TensorDataset(img)


    testloader = torch.utils.data.DataLoader(dataset=testset,
                                            batch_size=1,
                                            shuffle=False,
                                            drop_last=True)

    data = testloader.dataset[0][0]
    data = torch.unsqueeze(data, 0)

    output = model(data)
    print(output)

    pred = output.data.max(1, keepdim = True)[1]
    if pred[0][0] == 0:
        normal_cnt += 1
    else:
        broken_cnt += 1
    # print(pred[0][0])
    # print( str(normal_cnt) + "   " + str(broken_cnt))
    
# correct += pred.eq(target.data.view_as(pred)).cpu().sum()

# test_loss /= len(testloader.dataset)/batch_size

# print('\nTest set : Average loss : {: .4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(testloader.dataset), 100. * correct / len(testloader.dataset)))