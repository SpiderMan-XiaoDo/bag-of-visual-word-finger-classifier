import cv2
import numpy as np
import time
def find_bounder(image):
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    return magnitude


 
capture  = cv2.VideoCapture(0)
pTime = 0
while(True):
    _, img = capture.read()
    ycc = cv2.cvtColor(img , cv2.COLOR_BGR2YCR_CB)

    min_ycc = np.array([0,133,85], np.uint8)
    max_ycc = np.array([255,176,140], np.uint8 )
    skin  = cv2.inRange(ycc, min_ycc, max_ycc)

    cTime = time.time()
    fps = 1/ (cTime - pTime)
    pTime = cTime
    cv2.putText(img,f'FPS: {int(fps)}', (150, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3 )

    # img_gradient = find_bounder(skin)
    ret, thresh = cv2.threshold(skin, 200, 255, cv2.THRESH_BINARY)

    cv2.imshow('threshold', thresh)

    # Find contours in the binary image
    try:
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        largest_contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest_contour)

    # Find convexity defects of the largest contour
        defects = cv2.convexityDefects(largest_contour, cv2.convexHull(largest_contour, returnPoints=False))

        # Display the convex hull and convexity defects
        cv2.drawContours(skin, [largest_contour], 0, (0, 255, 0), 2)  # Green color for the largest contour
        cv2.drawContours(skin, [hull], 0, (0, 0, 255), 2)  # Red color for convex hull

        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(largest_contour[s][0])
            end = tuple(largest_contour[e][0])
            far = tuple(largest_contour[f][0])
            cv2.line(img, start, end, [0, 255, 0], 2)
            cv2.circle(img,far,5,[0,0,255],-1)
              # Green color for convexity defects
        # print('start: ', start)
        # print('end: ', end)

    except:
        pass
    cv2.imshow('Contours and Convexity Defects', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
capture.release()
cv2.destroyAllWindows()