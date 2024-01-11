import numpy as np
import cv2

if __name__ == '__main__':
    file_txt_path = r'baocao_computervision\test\hand_detection\train\test_bounder_716_jpg.rf.798fc0726ad1e529cf91fb9624a90d05.txt'
    image_path = r'baocao_computervision\test\hand_detection\train\test_bounder_716_jpg.rf.798fc0726ad1e529cf91fb9624a90d05.jpg'
    
    # position = (50, 50)  # Tọa độ (x, y) của góc trái trên của văn bản
    font = cv2.FONT_HERSHEY_SIMPLEX  # Loại font
    font_scale = 1  # Tỉ lệ kích thước font
    font_color = (255, 255, 255)  # Màu văn bản (trắng trong BGR)
    font_thickness = 2  # Độ d
    
    image = cv2.imread(image_path)
    # cv2.imshow('image', image)
    with open(file_txt_path) as f:
        content = f.readlines()
    # print('content: ', content)
    left_hand = content[1].split(' ')[1:]
    cor_left = []
    for item in left_hand:
        try:
            cor_left.append(float(item))
        except Exception as e:
            print('Loi')
    right_hand = content[0].split(' ')[1:]
    cor_right = []
    for item in right_hand:
        try:
            cor_right.append(float(item))
        except Exception as e:
            print('Loi')
    print('left: ', cor_left)
    print('right: ', cor_right)

    H, W,_ = image.shape

    x1 = int((cor_left[0] - 0.5 * cor_left[2]) * W)  
    y1 = int((cor_left[1] - 0.5 * cor_left[3]) * H) 
    x2 = int((cor_left[0] + 0.5 * cor_left[2]) * W)
    y2 = int((cor_left[1] + 0.5 * cor_left[3]) * H)
   

    x3 = int((cor_right[0] - 0.5 * cor_right[2]) * W)  
    y3 = int((cor_right[1] - 0.5 * cor_right[3]) * H)  
    x4 = int((cor_right[0] + 0.5 * cor_right[2]) * W)
    y4 = int((cor_right[1] + 0.5 * cor_right[3]) * H)

    cropped_left_hand_image = image[y1:y2+1, x1:x2+1]
    cv2.imshow('crop left: ', cropped_left_hand_image)
    cropped_right_hand_image = image[y3:y4+1, x3:x4+1]
    cv2.imshow('crop right: ', cropped_right_hand_image)
    cv2.imwrite('crop_left.jpg', cropped_left_hand_image)
    cv2.imwrite('crop_right.jpg', cropped_right_hand_image)

    
    cv2.rectangle(image, (x3, y3), (x4, y4), (0, 0, 255), 2)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(image, 'left', (x1 + 30, y1 - 30), font, font_scale, font_color, font_thickness)
    cv2.putText(image, 'right', (x3 + 30, y3 - 30), font, font_scale, font_color, font_thickness)
    cv2.putText(image, 'result: ', (W - int(W/5), 30), font, font_scale , font_color, font_thickness)



    cv2.imshow('result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()