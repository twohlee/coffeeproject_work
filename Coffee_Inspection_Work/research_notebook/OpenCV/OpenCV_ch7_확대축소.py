# Main Code

import cv2

src = cv2.imread('./data/insect_damage.jpg', cv2.IMREAD_COLOR)

height, width, channel = src.shape

# cv2.pyrUP(원본이미지, 결과 이미지 크기, 픽셀 외삽법)을 의미
# 픽셀 외삽법은 이미지를 확대 또는 축소할 경우, 영역 밖의 픽셀은 추정해서 값을 할당
dst = cv2.pyrUp(src, dstsize = (width*2, height*2), borderType=cv2.BORDER_DEFAULT)
print(dst.shape)

# cv2.pyrDown(원본이미지, 결과 이미지 크기, 픽셀 외삽법)을 의미
dst2 = cv2.pyrDown(src)
print(dst2.shape)

cv2.imshow('scr', src)
cv2.imshow('dst', dst)
cv2.imshow('dst2', dst2)
cv2.waitKey(0)
cv2.destroyAllWindows()