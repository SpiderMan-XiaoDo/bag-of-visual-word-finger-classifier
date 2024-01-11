import cv2
import numpy as np
import glob
import os
from tqdm import tqdm
import joblib

def load_images(base_path):
    print('loading images!!')
    bgr_images = []
    gray_images = []
    folder_list = os.listdir(base_path)
    labelWindex = []
    index = 0
    for item in folder_list:
        
        labels_file_path = os.path.join(base_path, item)

        labels_file_list = os.listdir(labels_file_path)
        for label in labels_file_list:
            
            hand_image_path = os.path.join(base_path, item, label)
            hand_image_list = os.listdir(hand_image_path)
            for image in tqdm(hand_image_list):
                image_path = os.path.join(hand_image_path, image)
                finger_index = int(label)
                # if(item == 'left'):
                #     finger_index*= 10
                # if item == 'left':
                #     finger_index = 1
                # else:
                #     finger_index = 2
                labelWindex.append((index, finger_index,))
                index+= 1
                bgr = cv2.imread(image_path)

                # Thay đổi đoạn code:

                # gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


                # Chuyển thành:
                converted2 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

                converted =  cv2.cvtColor(bgr , cv2.COLOR_BGR2YCR_CB)
                min_ycc = np.array([0,133,85], np.uint8)
                max_ycc = np.array([255,176,140], np.uint8 )
                skinMask = cv2.inRange(converted, min_ycc, max_ycc)
                # skinMask = cv2.addWeighted(skinMask,0.5,skinMask,0.5,0.0)             
                skinMask = cv2.medianBlur(skinMask, 5)

                
                skin = cv2.bitwise_and(converted2, converted2, mask = skinMask)
                # img2 = cv2.Canny(skin,60,60)
                bgr_images.append(bgr)
                gray_images.append(skin)
    print('loaded images!!')
    return bgr_images, gray_images, labelWindex



def extract_visual_features(gray_images):
# Extract SIFT features from gray images

# Sử dụng SIFT:
    # Define our feature extractor (SIFT)
    extractor = cv2.SIFT_create()
# Sử dụng SUFT
    # surf = cv2.SURF_create()
    keypoints = []
    descriptors = []
    print('extracting features!!')

    for img in tqdm(gray_images):
        # index+=1
        # if(index % 200 == 0):
            # print('the number of images processed: ', index)
        # extract keypoints and descriptors for each image

        # Thực hiện phát hiện keypoint và descriptors:
    # Bằng SIFT:
        
        img_keypoints, img_descriptors = extractor.detectAndCompute(img, None)
    # Bằng SUFT:
        # img_keypoints, img_descriptors = surf.detectAndCompute(img,None)
        if img_descriptors is not None:
            keypoints.append(img_keypoints)
            descriptors.append(img_descriptors)
    print('extracted features!!')
    return keypoints, descriptors


def visualize_keypoints(bgr_image, image_keypoints):
    cv2.drawKeypoints(bgr_image, image_keypoints, 0, (255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return bgr_image.copy()



