import cv2

# src_path = "./img/keyboard.bmp"
# out_path = "./img/keyboard_output.bmp"
#######################################3

# src_path = "./img/bean/1.jpg"

# ori = cv2.imread(src_path, cv2.IMREAD_COLOR)
# gray = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)

# cv2.imshow("Original", ori)
# cv2.imshow("Grayscale", gray)

# _, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
# cv2.imshow("dst", dst)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

for i in range(1,18):
    print(i)

    src_path = "./img/bean/" + str(i) + ".jpg"
    gray = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)

    _, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    cv2.imwrite("./img/bean_binary/"+ str(i) + ".jpg", dst)