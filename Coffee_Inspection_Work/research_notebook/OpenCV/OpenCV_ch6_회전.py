# Main Code

import cv2

src = cv2.imread('./data/insect_damage.jpg', cv2.IMREAD_COLOR)

height, width, channel = src.shape

# matrix에 회전 array를 생성
# cv2.getRotationMatrix2D((중심점 X좌표, 중심점 Y좌표), 각도, 스케일)을 설정
matrix = cv2.getRotationMatrix2D((width/2, height/2), 90 , 1)
# print(type(matrix)) => <class 'numpy.ndarray'>

# cv2.warpAffine(원본 이미지, 배열, (결과 이미지 너비, 결과 이미지 높이))을 의미
dst = cv2.warpAffine(src, matrix, (width, height))
# print(type(dst)) => <class 'numpy.ndarray'>



cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

import numpy as np
