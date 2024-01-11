import cv2
import numpy as np
import math
def find_bounder(image):
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Tính toán độ lớn của gradient
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    return magnitude

 
img = cv2.imread(r'baocao_computervision\dataset\train\right\8\399925675_343357038346438_3792388366878569319_n.jpg')
ycc = cv2.cvtColor(img , cv2.COLOR_BGR2YCR_CB)

min_ycc = np.array([0,133,85], np.uint8)
max_ycc = np.array([255,170,125], np.uint8 )
skin  = cv2.inRange(ycc, min_ycc, max_ycc)

# cv2.imshow('imagebf', img)
# cv2.imshow('imageaft', skin)
# img_gradient = find_bounder(skin)
# cv2.imshow('image bounder: ', img_gradient)


# bound_img = img.copy()
# print('skin: ', skin)
# print(type(bound_img))
# bound_img[img_gradient == 255] = [0, 255, 0]
# cv2.imshow('image_with_boxes', bound_img)
ret, thresh = cv2.threshold(skin, 100, 255, cv2.THRESH_BINARY)

cv2.imshow('threshold', thresh)

# Find contours in the binary image
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
cv2.waitKey(0)
cv2.destroyAllWindows()