# Contents
# 윤곽선이나 이미지의 0차 모멘트 부터 3차 모멘트까지 계산하는 알고리즘
# 공간모멘트, 중심모멘트, 정규환된 중심모멘트, 질량중심 등을 계싼


# Main Code

import cv2

src = cv2.imread('./data/insect_damage.jpg')
dst = src.copy()

gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
ret, binary = cv2.threshold(gray,150,255, cv2.THRESH_BINARY_INV)

contours, hierachy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

for i in contours:
    print(i)
    M = cv2.moments(i,False)
    cX = int(M['m10'] / M['m00'])
    cY = int(M['m01'] / M['m00'])

    cv2.circle(dst, (cX, cY), 3, (255,0,0), -1)
    cv2.drawContours(dst, [i], 0, (0,0,255), 2)

cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()