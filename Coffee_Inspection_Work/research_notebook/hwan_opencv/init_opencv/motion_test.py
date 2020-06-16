import numpy as np
import cv2

# fname = './image/fbi.avi'
# fname = './image/walking.gif'
# fname = './image/trump.gif'
fname = './image/trump2.gif'

cap = cv2.VideoCapture(fname)

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

while(1):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame',gray)

    fgmask = fgbg.apply(frame)
    cv2.imshow('frame',fgmask)


    k = cv2.waitKey(60) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()