# Main Code

import cv2

src = cv2.imread('./data/insect_damage.jpg', cv2.IMREAD_COLOR)


# 이진화를 적용하기 위해서 그레이스케일로 변환
gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
# ret : 임계값 저장
# cv2.threshold(그레이스케일 이미지, 임계값, 최댓값, 임계값 종류)
# 임계값 보다 작으면 0, 임계값 보다 크면 1
ret, dst = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)

cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()