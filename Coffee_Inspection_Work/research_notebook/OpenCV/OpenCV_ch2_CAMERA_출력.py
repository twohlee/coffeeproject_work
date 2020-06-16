# Camera 출력
# 내장 카메라 또는 외장카메라에서 이미지를 얻어와 프레임을 재생할 수 있다

# Main Code
import cv2

# 내장 카메라 또는 외장 카메라에서 영상을 받아온다
# cv2.VideoCapture(n), n은 카메라의 장치 번호
capture = cv2.VideoCapture(0)

# 카메라의 속성을 설정
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

# 영상 출력을 반복
while True:
    # ret : 정상 작동할 경우 True, 작동하지 않을 경우 False
    # frame에 현재 프레임이 저장
    ret, frame = capture.read()

    # 윈도우 창에 이미지를 띄움
    cv2.imshow('VideoFrame', frame)

    # cv2.waitKey(time), time마다 키 입력상태를 받아온다
    if cv2.waitKey(1) > 0 : break

# 카메라 장치에서 받아온 메모리를 헤제
capture.release()

# 모든 윈도우창을 닫는다.
# cv2.destroyAllWindows('윈도우 창 제목') => 특정 윈도우 창만 닫을 수 있다.
cv2.destroyAllWindows()
