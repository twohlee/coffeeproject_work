# OpecnCV 설치
# pip install opencv-python

# test
import cv2
import numpy as np

image = cv2.imread('./data/insect_damage.jpg', cv2.IMREAD_UNCHANGED)
cv2.imshow('insect_damage.jpg', image)
cv2.waitKey(0)
cv2.destroyAllWindows()