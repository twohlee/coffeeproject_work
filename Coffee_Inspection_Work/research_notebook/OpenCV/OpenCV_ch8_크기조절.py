# Main Code

import cv2

src = cv2.imread('./data/insect_damage.jpg', cv2.IMREAD_COLOR)

# cv2.resize(원본 이미지, 결과 이미지 크기, 보간법)
# 이미지의 크기를 변경하는 경우, 변형된 이미지의 픽셀은 추정해서 값을 할당
dst = cv2.resize(src, dsize = (640, 480), interpolation = cv2.INTER_AREA)

# cv2.resize(원본 이미지, dsize = (0,0), 가로비, 세로비, 보간법)로 이미지의 크기를 조절
dst2 = cv2.resize(src, dsize = (0,0), fx= 0.3, fy=0.7, interpolation=cv2.INTER_LINEAR)

# interpolation 속성
# cv2.INTER_NEAREST	이웃 보간법
# cv2.INTER_LINEAR	쌍 선형 보간법 => 가장 많이 사용 된다
# cv2.INTER_LINEAR_EXACT	비트 쌍 선형 보간법
# cv2.INTER_CUBIC	바이큐빅 보간법
# cv2.INTER_AREA	영역 보간법
# cv2.INTER_LANCZOS4	Lanczos 보간법

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.imshow('dst2', dst2)
cv2.waitKey(0)
cv2.destroyAllWindows()