import cv2 
import glob
import os
import sys
import numpy as np
from time import sleep, time

normal_cnt = 0
broken_cnt = 0

dataPath = "/home/team/project/GAN/bean/data"
normal_imgs = glob.glob(os.path.join(dataPath+"/normal_data/normal_third_processed_data",'*.jpg'))
normal_imgs = glob.glob(os.path.join(dataPath+"/broken_data/broken_third_processed_data",'*.jpg'))
# normal_imgs = glob.glob(os.path.join(dataPath+"/normal_data/normal_rotated_data",'*.jpg'))

for x in normal_imgs:
    img = cv2.imread(x, cv2.IMREAD_COLOR)
    imgray = cv2.imread(x, cv2.IMREAD_GRAYSCALE)
    bitwise = cv2.bitwise_not(imgray)
    _, thr = cv2.threshold(bitwise, 50, 255, cv2.THRESH_BINARY) 
    # 검은 배경 생성
    _, white = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY) 
    # black을 다시 3 channel로 변경 => 초록 타원을 씌우기 위함
    white = cv2.cvtColor(white, cv2.COLOR_GRAY2BGR)
    


    # cv2.imshow("img", img)
    # cv2.imshow("imgray", imgray)
    # cv2.imshow("black", black)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    _, contour, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    # contour 중에서 최대로 많은 것 탐색
    max_iter = 0
    max_len = 0
    for i in range(len(contour)):
        if len(contour[i]) > max_len:
            max_len = len(contour[i])
            max_iter = i
    # sleep(10)
    con = contour[max_iter] 
    (x, y), r = cv2.minEnclosingCircle(con) 
    # cv2.circle(img, (int(x), int(y)), int(r), (0, 0, 255), 2) 
    ellipse = cv2.fitEllipse(con) 
    # black에 ellipse를 그림
    cv2.ellipse(white, ellipse, (0, 255, 0), -1) 

    # cv2.imshow("white",white)
    # cv2.imshow("img",img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    start = time()
    check_num = 0
    white_mat = white[0][0]
    is_broken = False

    # normal => 12 정도 limit
    # broken => 들쭉날쭉 ..
    limit_cnt = 0
    for i in range(white.shape[0]):
        for j in range(white.shape[1]):
            r_number = white[i][j][0]
            if r_number == check_num and \
               img[i][j][0] == 255 and \
               img[i][j][1] == 255 and \
               img[i][j][2] == 255:
                is_broken = True
                limit_cnt += 1
                # break
        # if is_broken:
        #     break
    # print("end : " + str( time() - start ))
    # sleep(10)
    if is_broken:
        broken_cnt += 1
    else:
        normal_cnt += 1

    print("broken cnt : %d \nnormal cnt : %d"%(broken_cnt, normal_cnt))
    print("limit_cnt : " + str(limit_cnt))




    # cv2.imshow("black", black)
    # cv2.imshow('img', img) 
    # cv2.waitKey(0) 
    # cv2.destroyAllWindows()


    # 녹색으로 다 채운(-1)
    # 녹색 픽셀 획득
    # bgr:  [0, 255, 0]
    # hsv:  [ 60 255 255]
    # 녹색 픽셀 중 배경색(흰색)과 같은 픽셀이 있다 => 브로큰!!