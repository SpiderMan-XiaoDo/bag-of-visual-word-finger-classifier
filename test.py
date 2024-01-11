import os
import cv2
from tqdm import tqdm
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
                labelWindex.append((index, f'{item}_{label}',))
                index+= 1
                bgr = cv2.imread(image_path)
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                bgr_images.append(bgr)
                gray_images.append(gray)
    print('loaded images!!')
    return bgr_images, gray_images, labelWindex


bgr_images, gray_images, labelWindex = load_images(r'baocao_computervision\dataset\train')

for brg_image, gray_image in zip(bgr_images, gray_images):
    sift = cv2.SIFT_create()

    # Phát hiện keypoint và tính toán descriptors
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)

    # Vẽ các keypoint lên hình ảnh
    image_with_keypoints = cv2.drawKeypoints(brg_image, keypoints, None)

    # Hiển thị hình ảnh đã vẽ keypoint
    cv2.imshow('Image with Keypoints', image_with_keypoints)
    cv2.waitKey(0)
    cv2.destroyWindow('Image with Keypoints')
cv2.destroyAllWindows()