# Main Code

import cv2

src = cv2.imread('./data/insect_damage.jpg', cv2.IMREAD_COLOR)

# cv2.bitwise_not(원본 이미지)
dst = cv2.bitwise_not(src)

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()