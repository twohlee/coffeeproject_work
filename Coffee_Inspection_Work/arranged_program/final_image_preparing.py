# 해야하는거
# 1. normal img => 전처리 전부 다시해서 주변 노이즈 싹 제거
# 2. broken img => 전처리 전부 다시해서 주변 노이즈 싹 제거
# 3. black img => 전처리 전부 다시해서 주변 노이즈 싹 제거

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import torch
import glob


# 이미지 불러오기
# # normal
# path = './Final_data_rb/Normal/*.png'
# normal_name_list = glob.glob(path)
# normal_list = list()
# for img_normal in normal_name_list:
#     image = Image.open(img_normal)
#     normal_list.append(image)

# # broken
# path = './Final_data_rb/Broken/*.png'
# broken_name_list = glob.glob(path)
# broken_list = list()
# for img_broken in broken_name_list:
#     image = Image.open(img_broken)
#     broken_list.append(image)

# black
path = './Final_data_rb/Black/*.png'
black_name_list = glob.glob(path)
black_list = list()
for img_black in black_name_list:
    image = Image.open(img_black)
    black_list.append(image)

# 이미지 전처리함수
def imgPreprocessing(src, thres):
    ############################################################################
    # <이미지 노이즈 및 배경 제거>
    src_fitting = np.array(src)
    # 그레이 스케일로 변환
    gray = np.array(src.convert('L'))
    
    # 바이너리로 변환
    ret, binary = cv2.threshold(gray,thres,255, cv2.THRESH_BINARY)
    binary = cv2.bitwise_not(binary)

    # 바이너리 이미지와 원본 이미지의 데이터가 일치하는 부분 만 다시 추출히기
    tmp = np.zeros_like(src_fitting)
    for y in range(64):
        for x in range(64):
            if (binary != 0)[y,x]:
                for i in range(3):
                    tmp[y,x,i] = src_fitting[y,x,i]
    
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
    contours_leng = [len(i) for i in contours if len(i) != 252]
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

    return src, binary, src_processed, img_translation

# 이미지 생성 및 저장 => .png로 저장해야 한다
angle_list = [0, 30, 45, 60, 90 , 120, 135, 150, 180, 210, 225, 240, 270, 300, 315, 330]

width =  64
height = 64

items = [black_list]
item_names = ['black']
for idx0, item in enumerate(items):    
    for idx1, img in enumerate(item):
        # img_processed = Image.fromarray(imgPreprocessing(img,170)[3], 'RGB')
        # img_processed_final = imgPreprocessing(img,170)[3]
        img_processed_final = np.array(img)
        
        # plt.imshow(img_processed_final)
        # plt.show()
        cnt_sub = 0
        for idx2, angle in enumerate(angle_list):
            # matrix에 회전 array를 생성
            # cv2.getRotationMatrix2D((중심점 X좌표, 중심점 Y좌표), 각도, 스케일)을 설정
            matrix = cv2.getRotationMatrix2D((width/2, height/2), angle , 1)
            
            # cv2.warpAffine(원본 이미지, 배열, (결과 이미지 너비, 결과 이미지 높이))을 의미
            img_rotated = cv2.warpAffine(img_processed_final, matrix, (width, height))
            Image.fromarray(img_rotated).save('./Rotated_data_rb/' + item_names[idx0] + '/' + item_names[idx0] + '1st_pcd_' + str((idx1)*len(angle_list) + (idx2 + 1)) + '.png')

            print('cnt')




