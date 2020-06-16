# Main Code

import cv2

# cv2.imread('경로', mode)
image = cv2.imread('./data/insect_damage.jpg', cv2.IMREAD_ANYCOLOR)

print(image.shape) # (506, 900, 3) = (height, width, channel)


# mode
# cv2.IMREAD_UNCHANGED : 원본 사용
# cv2.IMREAD_GRAYSCALE : 1 채널, 그레이스케일 적용
# cv2.IMREAD_COLOR : 3 채널, BGR 이미지 사용
# cv2.IMREAD_ANYDEPTH : 이미지에 따라 정밀도를 16/32비트 또는 8비트로 사용
# cv2.IMREAD_ANYCOLOR : 가능한 3 채널, 색상 이미지로 사용
# cv2.IMREAD_REDUCED_GRAYSCALE_2 : 1 채널, 1/2 크기, 그레이스케일 적용
# cv2.IMREAD_REDUCED_GRAYSCALE_4 : 1 채널, 1/4 크기, 그레이스케일 적용
# cv2.IMREAD_REDUCED_GRAYSCALE_8 : 1 채널, 1/8 크기, 그레이스케일 적용
# cv2.IMREAD_REDUCED_COLOR_2 : 3 채널, 1/2 크기, BGR 이미지 사용
# cv2.IMREAD_REDUCED_COLOR_4 : 3 채널, 1/4 크기, BGR 이미지 사용
# cv2.IMREAD_REDUCED_COLOR_8 : 3 채널, 1/8 크기, BGR 이미지 사용


# cv2.imshow('윈도우 창 제목', 이미지)
cv2.imshow('insect_damage', image)
cv2.waitKey(0)
cv2.destroyAllWindows()