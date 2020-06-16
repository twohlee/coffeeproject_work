import cv2

fname = './image/coffee.png'

src = cv2.imread(fname, cv2.IMREAD_COLOR)

# cv2.resize( 원본 이미지, 결과 이미지 크기, interpolation(보간법) )
# 결과 이미지 크기 : Tuple형
dst = cv2.resize(src, dsize=(640, 480), interpolation=cv2.INTER_AREA)
# cv2.resize( 원본 이미지, dsize=(0, 0), 가로비, 세로비, interpolation(보간법) )
# 가로비, 세로비 => n배로 변경
dst2 = cv2.resize(src, dsize=(0, 0), fx=0.3, fy=0.7, interpolation=cv2.INTER_LINEAR)

cv2.imshow("src", src)
cv2.imshow("dst", dst)
cv2.imshow("dst2", dst2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# interpolation
# cv2.INTER_NEAREST      : 이웃 보간법
# cv2.INTER_LINEAR       : 쌍 선형 보간법, 주로 사용
# cv2.INTER_LINEAR_EXACT : 비트 쌍 선형 보간법
# cv2.INTER_CUBIC        : 바이큐빅 보간법
# cv2.INTER_AREA         : 영역 보간법
# cv2.INTER_LANCZOS4     : Lanczos 보간법
# 확대시, 바이큐빅 or 쌍선형
# 축소시, 영역 보간법