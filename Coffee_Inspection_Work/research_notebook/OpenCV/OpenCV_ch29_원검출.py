# Main Code

import cv2


# 그레이 스케일 이미지 사용
src = cv2.imread('./data/insect_damage.jpg')
dst = src.copy()
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)


# cv2.HoughCircles(검출 이미지, 검출 방법, 해상도 비율, 최소 거리, 캐니 엣지 임계값, 중심 임계값, 최소 반지름, 최대 반지름)
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100, param1 = 250, param2 = 60)

for i in circles[0]:
    cv2.circle(dst,(i[0], i[1]), i[2], (255,255,255), 5)

cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()