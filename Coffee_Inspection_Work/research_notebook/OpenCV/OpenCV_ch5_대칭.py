# Main Code

import cv2

src = cv2.imread('./data/20200508_test1_sample1.jpg', cv2.IMREAD_COLOR)

# cv2.flip(원본이미지, 대칭방법)
# 0 또는 0 이하 : 상하 대칭
# 1 또는 1 이상 : 좌우 대칭
dst = cv2.flip(src,0)

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()