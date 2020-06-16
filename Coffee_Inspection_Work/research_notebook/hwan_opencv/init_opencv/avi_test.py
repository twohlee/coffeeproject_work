from cv2 import cv2

# fname = './image/walking.gif'
fname = './image/fbi.avi'

cap = cv2.VideoCapture(fname)

while(cap.isOpened()):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame',gray)

    if cv2.waitKey(55) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()