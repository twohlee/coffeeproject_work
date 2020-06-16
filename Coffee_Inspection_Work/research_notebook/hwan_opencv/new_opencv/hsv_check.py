import cv2
import glob
import os
import sys
from time import sleep

dataPath = "/home/team/project/GAN/bean/data"
normal_imgs = glob.glob(os.path.join(dataPath+"/normal_rotated_data",'*.jpg'))
# normal_imgs = glob.glob(os.path.join(dataPath+"/broken_rotated_data",'*.jpg'))

for img in normal_imgs:

    img_color = cv2.imread(img)

    height,width = img_color.shape[:2]

    # img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)

    img_gray = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

    ret, thr = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY) 
    _, contour, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    con = contour[0] 
    (x, y), r = cv2.minEnclosingCircle(con) 
    ellipse = cv2.fitEllipse(con) 
    img_cover = img_color.copy()
    cv2.ellipse(img_cover, ellipse, (0, 255, 0), -1) 

    img_hsv = img_cover.copy()
    img_hsv = cv2.cvtColor(img_hsv, cv2.COLOR_BGR2HSV)
    
    cv2.imshow('img_color', img_color) 
    cv2.imshow('img_cover', img_cover) 
    cv2.imshow('img_hsv', img_hsv) 
    cv2.waitKey(0) 
    cv2.destroyAllWindows()


    lower_green = (60-10, 30, 30)
    upper_green = (60+10, 255, 255)
    img_mask = img_hsv.copy()
    img_mask = cv2.inRange(img_hsv, lower_green, upper_green)

    img_result = img_color.copy()
    img_result = cv2.bitwise_and(img_color, img_color, mask = img_mask)
    # print(img_result.shape)
    # sleep(10)

    cv2.imshow('img_color', img_color)
    cv2.imshow('img_mask', img_mask)
    cv2.imshow('img_result', img_result)


    cv2.waitKey(0)
    cv2.destroyAllWindows()