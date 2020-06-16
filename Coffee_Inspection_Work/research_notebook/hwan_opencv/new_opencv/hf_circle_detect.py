import numpy as np
import cv2
from time import sleep

img_gray = cv2.imread('./data/circle_test.jpg', cv2.IMREAD_GRAYSCALE)
# img_gray = cv2.imread('./data/normal/1.jpg', cv2.IMREAD_GRAYSCALE)


img_gray = cv2.medianBlur(img_gray,5)
img_color = cv2.cvtColor(img_gray,cv2.COLOR_GRAY2BGR)
cv2.imshow("img_gray", img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()

circles = cv2.HoughCircles(img_gray,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=35,minRadius=0,maxRadius=0)

try:
    circles = np.uint16(np.around(circles))
except:
    print("0 detect Circle!!")
    exit(0)

for c in circles[0,:]:

    center = (c[0],c[1])
    radius = c[2]
    
    # 바깥원
    cv2.circle(img_color,center,radius,(0,255,0),2)
    
    # 중심원
    cv2.circle(img_color,center,2,(0,0,255),9)

    cv2.imshow('detected circles',img_color)
    cv2.waitKey(0)
cv2.destroyAllWindows()