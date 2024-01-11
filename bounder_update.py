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

    height,width, _ = img.shape

    img1 = img[:int(height * 0.65), :int(width * 0.45), :]
    img2 = img[:int(height * 0.65), int(width * 0.55):, :]
    cv2.imshow('left: ', img1)
    cv2.imshow('right: ', img2)
    cTime = time.time()
    fps = 1/ (cTime - pTime)
    pTime = cTime
    cv2.putText(img,f'FPS: {int(fps)}', (150, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3 )

    for index, image in enumerate((img1, img2,)):
        ycc = cv2.cvtColor(image , cv2.COLOR_BGR2YCR_CB)
        
        min_ycc = np.array([0,133,85], np.uint8)
        max_ycc = np.array([255,170,125], np.uint8 )
        # min_ycc = np.array([0,133,85], np.uint8)
        # max_ycc = np.array([255,170,125], np.uint8)
        # min_ycc = np.array([0,133,97], np.uint8)
        # max_ycc = np.array([255,172,142], np.uint8 )
        # min_ycc = np.array([0,98,85], np.uint8)
        # max_ycc = np.array([255,142,125], np.uint8 )
        skin  = cv2.inRange(ycc, min_ycc, max_ycc)
        skin_copy = np.zeros((height, width), dtype=np.uint8)
        if(index == 0):
            skin_copy[:int(height * 0.65), : int(width * 0.45)] = skin
        else:
            skin_copy[:int(height * 0.65), int(width * 0.55):] = skin
        # img_gradient = find_bounder(skin_copy)
        cv2.imshow('bef: ', skin_copy)
        thresh = skin_copy.copy()
        # ret, thresh = cv2.threshold(skin_copy, 200, 255, cv2.THRESH_BINARY)
        cv2.imshow('threshold: ', thresh)
        try:
            contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            # contours, hierarchy = cv2.findContours(thresh,2,1)
            max_area = 0
            for i in range(len(contours)):
                cnt=contours[i]
                area = cv2.contourArea(cnt)
                if(area>max_area):
                    max_area=area
                    ci=i
            cnt=contours[ci]
            hull = cv2.convexHull(cnt)
            defects = cv2.convexityDefects(cnt, cv2.convexHull(cnt, returnPoints=False))
            # print('cnt: ', cnt.shape)

            for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]
                far = tuple(cnt[f][0])
                cv2.circle(img,far,5,[0,255,0],-1)
            
            # cv2.drawContours(img,cnt,0,(0,255,0),2)
            cv2.drawContours(img,[hull],0,(0,0,255),2)
            cv2.imshow('Contours and Convexity Defects', img)

        except Exception as e:
            print('loi: ', e)
    cv2.imshow('Contours and Convexity Defects', img)    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
capture.release()
cv2.destroyAllWindows()