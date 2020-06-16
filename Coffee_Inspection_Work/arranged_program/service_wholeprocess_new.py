# 필요한 모듈 불러오기
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
import cv2
import matplotlib.pyplot as plt
from time import sleep
import pickle
import json

# 환경 설정값
# shot = 1
thres = 180 # 이미지전처리 시, threshold 값


# =============================================================================================
# <데이터베이스에서 데이터 로드>
# conn = pymongo.MongoClient('192.168.0.113', 27017)
# db = conn.get_database('Raspberrypi')
# fs = gridfs.GridFS(db, collection = 'logdata')
# f = list(fs.find())[-1]
# img = Image.open(io.BytesIO(f.read()))


# <로컬에서 데이터 로드>
img = Image.open('./Models/20200604_test_three12_.png')


# <라즈베리파이 공유폴더에서 데이터 로드>
# img = Image.open('/run/user/1000/gvfs/smb-share:server=192.168.0.30,share=pi/logdata/20200604_test_three12_.png')

# =============================================================================================


def multi_objectDetection(img):
    img_color = np.array(img)
    img = np.array(img.convert('L'))

    # img_color_ROI = img_color[733:1437,710:1414]
    # img_ROI = img[733:1437,710:1414]
    # Take29까지
    # img_color_ROI = img_color[400:1700,400:1700]
    # img_ROI = img[400:1700,400:1700]
    # Take 30부터
    img_color_ROI = img_color[250:1550,200:1500]
    img_ROI = img[250:1550,200:1500]


    # 데이터 이진화
    _, src_bin = cv2.threshold(img_ROI, thres, 255, cv2.THRESH_BINARY)
    src_bin = cv2.bitwise_not(src_bin)

    # 데이터 처리
    img_list = list()
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(src_bin)

    # print(stats)

    alpha = 2
    SIZE = 128
    for img_num in range(1,nlabels):
        stat = stats[img_num]
        x = stat[0] - alpha
        y = stat[1] - alpha
        width = stat[2] + 2*alpha
        height = stat[3] + 2*alpha
        n_pixel = stat[4]

        
        # 최대 픽셀 지정해서 임계값보다 크면 패스해버리기
        # if n_pixel < 500 or width > 64 or height > 64 : continue
        if n_pixel < 3000 : continue

        # cv2.rectangle(img_color_ROI, (x, y), (x+width, y+height), (0,100,255))
        
        # # 사각박스에 레이블 달아주기
        # text = str(img_num)
        # cv2.putText(img_color_ROI, text=text, org=(x+int(width/2), y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=2)
        
        
        delta_x = int((SIZE - width) / 2)
        delta_y = int((SIZE - height) / 2)
        # print(delta_x, delta_y)
        
        
        tmp1 = img_color_ROI[y : y+height, x : x+width, :].copy()
        tmp2 = np.zeros((SIZE,SIZE,3), dtype = int)
        
        # print(stat) # 가장자리쪽에 있는 생두들은 ROI를 벗어나는것들이 있다.
        try:
            for channel in range(3):
                tmp2[delta_y : delta_y + height, delta_x : delta_x + width , channel] = tmp1[0:height, 0:width, channel]
            tmp2 = np.uint8(tmp2)
            img_list.append((img_num, stat, tmp2))
        except:
            print(img_num, '에러다임마')

        
        # img_list.append((img_num, stat, tmp2))
    return img_list, img_color_ROI
    
# =============================================================================================
# < 이미지 전처리 과정 >

def imgPreprocessing(src, thres, SIZE):
    ############################################################################
    # <이미지 노이즈 및 배경 제거>

    # 그레이 스케일로 변환
    gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    
    # 바이너리로 변환
    ret, binary = cv2.threshold(gray,thres,255, cv2.THRESH_BINARY)
    binary = cv2.bitwise_not(binary)

    # 바이너리 이미지와 원본 이미지의 데이터가 일치하는 부분 만 다시 추출히기
    tmp = np.zeros_like(src)
    for y in range(SIZE):
        for x in range(SIZE):
            if (binary != 0)[y,x]:
                for i in range(3):
                    tmp[y,x,i] = src[y,x,i]
    
    src_processed = tmp   

    # 이미지 크기 정의
    height = src_processed.shape[0]
    width = src_processed.shape[1]

    ############################################################################
    # <생두 무게 중심 구하기>
    R = list()
    for y in range(height):
        for x in range(width):
            if binary[y,x]:            
                R.append([y, x])

    # 질량의 합
    M = len(R)

    R = np.array(R)
    R_x = R[:,1]
    R_y = R[:,0]

    R_x_sum = R_x.sum()
    R_y_sum = R_y.sum()

    center = np.round(R_x_sum/M) , (np.round(R_y_sum/M))

    height_center = center[0]
    width_center = center[1]

    # print('center:', height_center, width_center)

 
    ############################################################################
    # <객체 외부의 노이즈 데이터 처리하기>

    # 바이너리의 윤곽선 추출
    _, contours, hierachy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)    
    
    # 생두 객체 윤곽과 윤곽벡터 구하기
    contours_leng = [len(i) for i in contours]
    contours_leng.sort()
    contours_leng = contours_leng[:-2]
    len_max = np.array(contours_leng).max()
    contour_vector = list()

    for i in contours:
        if len(i) == len_max:
            for point in i:
                contour_vector.append((point[0][1]-height_center, point[0][0]-width_center))
            
            # 윤곽 벡터 사이즈와 최대 사이즈 구하기
            vector_size = np.sqrt(np.array(contour_vector)[:,0]**2 + np.array(contour_vector)[:,1]**2 )
            max_size = vector_size.max()
    

    # 질량 벡터와 윤곽벡터를 비교하는 코드 짜기
    R_size = np.sqrt((R[:,1] - width_center) ** 2 + (R[:,0] - height_center) ** 2)
    R_total = np.hstack([R,R_size.reshape((R_size.shape[0],1))])

    # max_size 보다 큰 R 좌표들만 추출하기
    R_filtered = R_total[R_total[:,2] > max_size]


    # R_filtered의 좌표를 이용해서 노이즈 처리
    for i in R_filtered[:,:2]:
        src_processed[int(i[1]),int(i[0])] = 0

    ############################################################################
    # <이미지의 중심과 생두의 무게중심 일치시키기>
    
    # 1. 이미지 센터 좌표를 구한다 
    src_center = np.array([src_processed.shape[1] / 2, src_processed.shape[0] / 2])

    # 2. 객체의 중심을 구한다
    object_center = np.array([height_center, width_center])

    # 3. 이미지 중심과 객체의 중심의 차이를 구한다
    delta = object_center - src_center

    # 4. 이미지 이동
    height, width = src_processed.shape[:2]
    M = np.float32([[1, 0, -delta[0]], [0, 1, -delta[1]]]) # 이미지를 width 방향으로 delta[0]만큼 , hiehgt 방향으로 -delta[1]만큼
    img_translation = cv2.warpAffine(src_processed, M, (width,height))

    # 5. 이미지 축소 ( 128 x 128 ) -> ( 64, 64 )
    
    height, width, channel = img_translation.shape
    img_constraction = cv2.pyrDown(img_translation)
    

    return src, binary, src_processed, img_translation, contours_leng, img_constraction

# ====================================================================================================================================
# <라즈베리파이 이미지 -> 128 x 128 이미지를 리스트에 저장>
img_list = multi_objectDetection(img)[0]


img_processed_list = list() # [128 x 128 이미지, 64 x 64 이미지]
for num in range(len(img_list)):
    try:
        img_processed = imgPreprocessing(img_list[num][2], thres, 128)[3]
        img_constraction = imgPreprocessing(img_list[num][2], thres, 128)[5]
        img_processed_list.append([img_processed, img_constraction])
        # Image.fromarray(img_constraction).save('./Final_data_rb/Black/black_shot' + str(shot) + '_' + str(num) +'.png' )
    except:
        print(num, '또 에러다')








# ====================================================================================================================================
# < 모델 로드 >
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
# model_path = "./Models/test_model.pth" # basic CNN
model_path = "./Models/test_model_crossValidation.pth" # 교차검증
model = CNN().to(device)
model.load_state_dict(torch.load(model_path, map_location = torch.device(device)))
model.eval()



# 모델 예측
# < 생두 분류 예측 >
prediction = list()
for i in range(len(img_processed_list)):


    # img = torch.Tensor(np.array(Image.open(img_path).convert("L")))
    image = torch.Tensor(np.array(Image.fromarray(img_processed_list[i][1]).convert("L")))
    image = torch.unsqueeze(image, 0)
    image = torch.unsqueeze(image, 0)
    image = image.to(device)

    testset = TensorDataset(image)
    testloader = torch.utils.data.DataLoader(dataset=testset,
                                            batch_size=1,
                                            shuffle=False,
                                            drop_last=True)

    data = testloader.dataset[0][0]
    data = torch.unsqueeze(data, 0)
    output = model(data)
    pred = output.data.max(1, keepdim = True)[1]
    
    # 세가지 분류 기준
    if pred[0][0] == 0:
        prediction.append((pred[0][0], 'Nr'))
    elif pred[0][0] == 1:
        prediction.append((pred[0][0], 'Br'))
    else:
        prediction.append((pred[0][0], 'Bl'))
        
    # # 정상 비정상 분류
    # if pred[0][0] == 0:
    #     prediction.append((pred[0][0], 'Nr'))
    # else :
    #     prediction.append((pred[0][0], 'ANr'))

prediction = np.array(prediction)
# prediction = np.array([ (x,'N') for x in range(len(img_processed_list))])



# ====================================================================================================================================
# 촬영 원본 이미지에 레이블링 하기
def labeledImage(img, prediction, img_list):
    img_color = np.array(img.convert("RGB"))
    img = np.array(img.convert('L'))
    img_color_ROI = img_color[250:1550,200:1500]
    img_ROI = img[250:1550,200:1500]

    # 데이터 이진화
    _, src_bin = cv2.threshold(img_ROI, thres, 255, cv2.THRESH_BINARY)
    src_bin = cv2.bitwise_not(src_bin)

    # 데이터 처리
    alpha = 2
    SIZE = 128
    for idx in range(len(img_processed_list)):
        stat = img_list[idx][1]
        x = stat[0] - alpha
        y = stat[1] - alpha
        width = stat[2] + 2*alpha
        height = stat[3] + 2*alpha
        n_pixel = stat[4]
    
        # 최대 픽셀 지정해서 임계값보다 크면 패스해버리기
        if n_pixel < 3000: continue
        
        cv2.rectangle(img_color_ROI, (x, y), (x+width, y+height), (0,100,255), thickness= 3)
        
        # 사각박스에 레이블 달아주기
        text = str(prediction[idx,1]) #(img_num , label) 
        cv2.putText(img_color_ROI, text=text, org=(x+int(width/2), y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=3)
    return img_color_ROI

# ====================================================================================================================================
# 최종 산출물
# 1. img (촬영된 이미지)
fig = plt.figure(figsize = (8,8))
plt.imshow(np.array(img)[250:1550,200:1500])
plt.show()
# img.save('/run/user/1000/gvfs/smb-share:server=192.168.0.30,share=pi/static/result/img.png')
img.save('./web/static/result/img.png')



# 2. prediction_for_web (prediction에서 종류별로 카운팅)
with open('./web/static/result/prediction_for_web.pickle', 'wb') as f:
    prediction_for_web = json.dumps({
        "Normal_cnt" : np.count_nonzero(prediction[:,1] == 'Nr'),
        "Broken_cnt" : np.count_nonzero(prediction[:,1] == 'Br'),
        "Black_cnt" : np.count_nonzero(prediction[:,1] == 'Bl')
    })
    pickle.dump(prediction_for_web, f)
    

# 3. labeledImage(img, prediction, img_list) (img 파일이 전처리 되어 표식되어 나온 이미지)
img_labeled = Image.fromarray(labeledImage(img, prediction, img_list))
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.imshow(img_labeled)
plt.show()
img_labeled.save('./web/static/result/img_labeled.png')







# ====================================================================================================================================
print('객체인식된 생두 개수: ', len(img_list))
print('전처리완료된 생두 개수: ', len(img_processed_list))
print('전처리 중에 오류난 생두 개수: ',  len(img_list) - len(img_processed_list))



# 이미지 확인(코드 작동과 무관)



# fig = plt.figure(figsize = (15,8))
# for idx in range(len(img_list)):
#     ax = fig.add_subplot(1,len(img_list), idx+1)
#     ax.imshow(img_list[idx][2])

# fig = plt.figure(figsize = (15,8))
# for idx in range(len(img_processed_list)):
#     ax = fig.add_subplot(1,len(img_processed_list), idx+1)
#     ax.imshow(img_processed_list[idx][1])


