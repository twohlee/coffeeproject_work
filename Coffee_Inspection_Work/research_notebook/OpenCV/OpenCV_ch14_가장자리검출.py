# Main Code

import cv2

src = cv2.imread('./data/insect_damage.jpg', cv2.IMREAD_COLOR)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

# cv2.Canny(원본 이미지, 임계값1, 임계값2, 커널 크기, L2그라디언터)를 이용하여 가장자리 검출을 적용
# 임계값 1 이하에 포함된 가장자리는 가장자리에서 제외
# 임계값 2 이상에 포함된 가장자리는 가장자리로 간주
# 커널크기 : Sobel 마스크의 Aperture Size를 의미
# L2그라디언트 : sqrt((dI/dx)2+(dI/dy)2)
# L1그라디언트 : ∥dI/dx∥+∥dI/dy∥
canny = cv2.Canny(src, 100, 255)

# cv2.Sobel(그레이스케일 이미지, 정밀도, x방향 미분, y방향 미분, 커널, 배율, 델타, 픽셀 외삽법)
# 커널 : sobel 커널의 크기를 선정 1, 3, 5, 7 값을 사용
# 배율 : 계산된 미분 값에 대한 배율값
# 델타 : 계산전 미분 값에 대한 추가값
sobel = cv2.Sobel(gray, cv2.CV_8U, 1,0,3)

# cv2.Laplacian(그레이스케일 이미지, 정밀도, 커널, 배율, 델타, 픽셀 외삽법)
laplacian = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)

# 픽셀 외삽법 종류
# 속성	의미
# cv2.BORDER_CONSTANT	iiiiii | abcdefgh | iiiiiii
# cv2.BORDER_REPLICATE	aaaaaa | abcdefgh | hhhhhhh
# cv2.BORDER_REFLECT	fedcba | abcdefgh | hgfedcb
# cv2.BORDER_WRAP	cdefgh | abcdefgh | abcdefg
# cv2.BORDER_REFLECT_101	gfedcb | abcdefgh | gfedcba
# cv2.BORDER_REFLECT101	gfedcb | abcdefgh | gfedcba
# cv2.BORDER_DEFAULT	gfedcb | abcdefgh | gfedcba
# cv2.BORDER_TRANSPARENT	uvwxyz | abcdefgh | ijklmno
# cv2.BORDER_ISOLATED	관심 영역 (ROI) 밖은 고려하지 않음

cv2.imshow('canny', canny)
cv2.imshow('sobel', sobel)
cv2.imshow('laplacian', laplacian)
cv2.waitKey(0)
cv2.destroyAllWindows()

