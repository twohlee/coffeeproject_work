import cv2
import time
white_fname = './image/white.jpg'
# 1~9
# fnum = 1

for fnum in range(0,10):
    fname = './mnist_image/{}.png'.format(fnum)
    output_resize_fname  = './mnist_image/{}.png'.format(fnum)
    output_contour_fname = './mnist_image/{}_contour.png'.format(fnum)
    output_inside_fname = './mnist_image/{}_inside.png'.format(fnum)

    white_contour = cv2.imread(white_fname , cv2.IMREAD_COLOR) # 윤곽선 배경
    white_contour_resize = cv2.resize(white_contour, dsize=(28, 28), interpolation=cv2.INTER_AREA) # resize
    white_inside = cv2.imread(white_fname , cv2.IMREAD_COLOR) # 윤곽선 배경
    white_inside_resize = cv2.resize(white_inside, dsize=(28, 28), interpolation=cv2.INTER_AREA) # resize
    src = cv2.imread(fname , cv2.IMREAD_COLOR) # 원본 이미지
    # src_resize = cv2.resize(src, dsize=(28, 28), interpolation=cv2.INTER_AREA) # resize
    gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY) # contour용

    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # binary = cv2.bitwise_not(binary)
    # cv2.imshow("gray", gray)
    cv2.imshow("binary", binary)

    contours, hierachy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    max_iter = 0
    max = 0
    for i in range(len(contours)):
        if max < len(contours[i]):
            max = len(contours[i])
            max_iter = i
    # print(len(contours))

    cv2.drawContours(src, [contours[max_iter]], 0, (0, 0, 255), thickness=1)
    cv2.imshow("src_resize", src)
    cv2.drawContours(white_contour_resize, [contours[max_iter]], 0, (0, 0, 255),thickness=1  )
    cv2.imshow("white_contour", white_contour_resize)
    cv2.drawContours(white_inside_resize, [contours[max_iter]], 0, (0, 0, 255),thickness=-1  )
    cv2.imshow("white_inside", white_inside_resize)

    # cv2.waitKey(0)
    cv2.imwrite(output_contour_fname, white_contour_resize)
    cv2.imwrite(output_inside_fname, white_inside_resize)

    cv2.destroyAllWindows()
