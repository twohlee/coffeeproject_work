import cv2

img_color = cv2.imread('./data/hsv_test.jpg')
height,width = img_color.shape[:2]

img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
cv2.imshow('img_hsv', img_hsv)
cv2.waitKey(0) 
cv2.destroyAllWindows()


lower_blue = (60-10, 30, 30)
upper_blue = (60+10, 255, 255)
img_mask = cv2.inRange(img_hsv, lower_blue, upper_blue)


img_result = cv2.bitwise_and(img_color, img_color, mask = img_mask)


cv2.imshow('img_color', img_color)
cv2.imshow('img_mask', img_mask)
cv2.imshow('img_result', img_result)
cv2.waitKey(0) 
cv2.destroyAllWindows()