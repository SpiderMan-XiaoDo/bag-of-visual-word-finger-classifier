import cv2
import numpy as np
import glob
import os
from tqdm import tqdm
import joblib
def load_images(base_path):
    print('loading images!!')
    folder_list = os.listdir(base_path)
    for item in folder_list:
        
        labels_file_path = os.path.join(base_path, item)

        labels_file_list = os.listdir(labels_file_path)
        for label in labels_file_list:
            
            hand_image_path = os.path.join(base_path, item, label)
            hand_image_list = os.listdir(hand_image_path)
            for image in tqdm(hand_image_list):
                image_path = os.path.join(hand_image_path, image)
                bgr = cv2.imread(image_path)
                cv2.imshow("original",bgr)

                converted2 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

                converted = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
                cv2.imshow('hsv: ', converted)
                # Convert from RGB to HSV
                #cv2.imshow("original",converted2)

                lowerBoundary = np.array([0,40,30],dtype="uint8")
                upperBoundary = np.array([43,255,254],dtype="uint8")
                skinMask = cv2.inRange(converted, lowerBoundary, upperBoundary)
                skinMask = cv2.addWeighted(skinMask,0.5,skinMask,0.5,0.0)
                cv2.imshow("masked",skinMask)
                
                skinMask = cv2.medianBlur(skinMask, 5)
                cv2.imshow("masked blur",skinMask)

                
                skin = cv2.bitwise_and(converted2, converted2, mask = skinMask)
                #frame = cv2.addWeighted(frame,1.5,skin,-0.5,0)
                #skin = cv2.bitwise_and(frame, frame, mask = skinMask)

                #skinGray=cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)
                
                cv2.imshow("masked2",skin)
                img2 = cv2.Canny(skin,60,60)
                cv2.imshow('image', img2)
                cv2.waitKey(0)
    print('loaded images!!')


def load_images_img2(base_path):
    print('loading images!!')
    folder_list = os.listdir(base_path)
    extractor = cv2.SIFT_create()
    for item in folder_list:
        
        labels_file_path = os.path.join(base_path, item)

        labels_file_list = os.listdir(labels_file_path)
        for label in labels_file_list:
            
            hand_image_path = os.path.join(base_path, item, label)
            hand_image_list = os.listdir(hand_image_path)
            for image in tqdm(hand_image_list):
                image_path = os.path.join(hand_image_path, image)
                bgr = cv2.imread(image_path)
                cv2.imshow("original",bgr)

                converted2 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

                converted =  cv2.cvtColor(bgr , cv2.COLOR_BGR2YCR_CB)
                min_ycc = np.array([0,133,85], np.uint8)
                max_ycc = np.array([255,176,140], np.uint8 )
                skinMask = cv2.inRange(converted, min_ycc, max_ycc)
                skinMask = cv2.addWeighted(skinMask,0.5,skinMask,0.5,0.0)
                cv2.imshow("masked",skinMask)      
                skinMask = cv2.medianBlur(skinMask, 5)
                cv2.imshow("masked blur",skinMask)

                
                skin = cv2.bitwise_and(converted2, converted2, mask = skinMask)     
                cv2.imshow("masked2",skin)
                img_keypoints, img_descriptors = extractor.detectAndCompute(skin, None)
                img2 = cv2.drawKeypoints(skin,img_keypoints,None,(0,0,255),4)
                
                cv2.imshow('image', img2)

                cv2.waitKey(0)
    print('loaded images!!')
load_images_img2(base_path=r'baocao_computervision\dataset _v2\train')