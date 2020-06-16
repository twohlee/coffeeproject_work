import cv2 
import glob
import os
import sys
import numpy as np
from time import sleep, time

normal_cnt = 0
broken_cnt = 0

dataPath = "/home/team/project/GAN/bean/data"
normal_imgs = glob.glob(os.path.join(dataPath+"/normal_data/normal_rotated_data",'*.jpg'))
# normal_imgs = glob.glob(os.path.join(dataPath+"/broken_data/broken_rotated_data",'*.jpg'))
for x in normal_imgs:
    img = cv2.imread(x, cv2.IMREAD_COLOR)
    imgray = cv2.imread(x, cv2.IMREAD_GRAYSCALE)
    # imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    _, thr = cv2.threshold(imgray, 50, 255, cv2.THRESH_BINARY) 

    # 검은 배경 생성
    _, black = cv2.threshold(imgray, 254, 255, cv2.THRESH_BINARY) 
    # black을 다시 3 channel로 변경 => 초록 타원을 씌우기 위함
    black = cv2.cvtColor(black, cv2.COLOR_GRAY2BGR)

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
    cv2.ellipse(black, ellipse, (0, 255, 0), -1) 

    start = time()
    check_num = 255
    black_mat = black[0][0]
    is_broken = False

    # normal => 이상치 빼고 1자리 초반으로 잘 나오는 듯
    # broken => 1자리 수가 생각보다 많다.
    limit_cnt = 13
    for i in range(black.shape[0]):
        for j in range(black.shape[1]):
            g_number = black[i][j][1]
            if g_number == check_num and \
               img[i][j][0] == 0 and \
               img[i][j][1] == 0 and \
               img[i][j][2] == 0:
                limit_cnt -= 1
                if limit_cnt == 0:
                    is_broken = True
                    break
        if is_broken:
            break
    # print("end : " + str( time() - start ))
    # sleep(10)
    if is_broken:
        broken_cnt += 1
    else:
        normal_cnt += 1

    print("broken cnt : %d \n normal cnt : %d"%(broken_cnt, normal_cnt))
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

    # normal 
    # broken cnt : 107 
    # normal cnt : 13445

    # broken
    # broken cnt : 1662 
    # normal cnt : 11890