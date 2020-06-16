import cv2

# src_path = "./img/keyboard.bmp"
# out_path = "./img/keyboard_output.bmp"

src_path = "./img/dog.jpeg"
out_path = "./img/dog_output.jpeg"

ori = cv2.imread(src_path, cv2.IMREAD_COLOR)
gray = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)

cv2.imshow("Original", ori)
cv2.imshow("Grayscale", gray)


#############
def onChange(x):
    pass

cv2.namedWindow('dst',cv2.WINDOW_NORMAL)
cv2.createTrackbar('Threshold', 'dst',0,255, onChange)
cv2.setTrackbarPos('Threshold', 'dst',128)

while True:
    thold = cv2.getTrackbarPos('Threshold', 'dst')
    _, dst = cv2.threshold(gray, thold, 255, cv2.THRESH_BINARY)

    cv2.imshow('dst', dst)

    k = cv2.waitKey(1)
    if k == 27:
        break
#############


cv2.waitKey(0)
cv2.destroyAllWindows()