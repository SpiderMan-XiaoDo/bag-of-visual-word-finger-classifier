import numpy as np
import cv2

backSub = cv2.createBackgroundSubtractorKNN(0, 100)
capture = cv2.VideoCapture(0)

while(True):
    _, frame = capture.read()
    mask = backSub.apply(frame)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    # cv2.imshow('origin', frame)
    # cv2.imshow('mask', res)
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    ret, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
    # cv2.imshow('blur', thresh)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    print(contours)
    print(hierarchy)
    
    if contours and len(contours[0]) >= 3:
        hull = cv2.convexHull(contours[0])
        
        if len(hull) > 5:
            # pass  # Đảm bảo hull có ít nhất 1 điểm
            defects = cv2.convexityDefects(contours[0], hull)
            cv2.drawContours(frame, [hull], 0, (0, 255, 0), 2)
            cv2.imshow('frame', frame)
    if cv2.waitKey(1) == 13:
        break
capture.release()
cv2.destroyAllWindows()