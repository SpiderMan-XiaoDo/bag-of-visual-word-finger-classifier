import numpy as np
import cv2
import bovw
import joblib
def load_features(file_path):
    print('loading file')
    print('loaded')
    return joblib.load(file_path)


def calc_codinate():
    file_txt_path = r'baocao_computervision\test\hand_detection\train\test_bounder_716_jpg.rf.798fc0726ad1e529cf91fb9624a90d05.txt'
    image_path = r'baocao_computervision\test\hand_detection\train\test_bounder_716_jpg.rf.798fc0726ad1e529cf91fb9624a90d05.jpg'
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
    # print('left: ', cor_left)
    # print('right: ', cor_right)

    H, W,_ = image.shape

    x1 = int((cor_left[0] - 0.5 * cor_left[2]) * W)  
    y1 = int((cor_left[1] - 0.5 * cor_left[3]) * H) 
    x2 = int((cor_left[0] + 0.5 * cor_left[2]) * W)
    y2 = int((cor_left[1] + 0.5 * cor_left[3]) * H)
   

    x3 = int((cor_right[0] - 0.5 * cor_right[2]) * W)  
    y3 = int((cor_right[1] - 0.5 * cor_right[3]) * H)  
    x4 = int((cor_right[0] + 0.5 * cor_right[2]) * W)
    y4 = int((cor_right[1] + 0.5 * cor_right[3]) * H)

    return x1, y1, x2, y2, x3, y3, x4, y4



if __name__ == '__main__':
    # Định nghĩa file path
    videopath = r'baocao_computervision\test\video_test\video_test.mp4'
    codebook_model_path = r'baocao_computervision\model\codebook\codebook_1000.pkl'

    svm_model_PCA_path = r'baocao_computervision\model\smv_pca\svm_model_PCA_230d.pkl'
    pca_model_path = r'baocao_computervision\model\smv_pca\pca_model.pkl'
    scaler_model_path = r'baocao_computervision\model\smv_pca\scaler.pkl'


    n_centroids = bovw.load_codebook(codebook_model_path)
    pca = load_features(pca_model_path)
    scaler = load_features(scaler_model_path)
    pca_svm_model = load_features(svm_model_PCA_path)
    # Định nghĩa các font chữ:
    font = cv2.FONT_HERSHEY_SIMPLEX  # Loại font
    font_scale = 1  # Tỉ lệ kích thước font
    font_color = (255, 255, 255)  # Màu văn bản (trắng trong BGR)
    font_thickness = 2  # Độ d


    capture = cv2.VideoCapture(videopath)
    # Get the frame rate of the video
    fps = capture.get(cv2.CAP_PROP_FPS)
    # index = 0
    while capture.isOpened():
        ret, frame = capture.read()

        if not ret:
            print("Error reading the video feed.")
            break
        x1, y1, x2, y2, x3, y3, x4, y4 = calc_codinate()

        converted2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        converted =  cv2.cvtColor(frame , cv2.COLOR_BGR2YCR_CB)
        min_ycc = np.array([0,140,85], np.uint8)
        max_ycc = np.array([255,176,140], np.uint8 )
        skinMask = cv2.inRange(converted, min_ycc, max_ycc)
        skinMask = cv2.addWeighted(skinMask,0.5,skinMask,0.5,0.0)             
        skinMask = cv2.medianBlur(skinMask, 5)


        H, W, _ = frame.shape

        # Thực hiện trích xuất đặc trưng và dự đoán:
        skin = cv2.bitwise_and(converted2, converted2, mask = skinMask)
        cropped_left_hand_image = skin[y1:y2+1, x1:x2+1]
        # cv2.imshow('crop left: ', cropped_left_hand_image)
        cropped_right_hand_image = skin[y3:y4+1, x3:x4+1]
        # cv2.imshow('crop right: ', cropped_right_hand_image)
            # Dự đoán:
    
        extractor = cv2.SIFT_create()
        left_hand_keypoints, left_hand_descriptors = extractor.detectAndCompute(cropped_left_hand_image, None)
        right_hand_keypoints, right_hand_descriptors = extractor.detectAndCompute(cropped_right_hand_image, None)

       
        if left_hand_descriptors is None or not left_hand_descriptors.any():
            print('None')
        elif right_hand_descriptors is None or not right_hand_descriptors.any():
            print('None')
        else:
            left_hand_vector = bovw.represent_image_features(left_hand_descriptors, n_centroids)
            right_hand_vector = bovw.represent_image_features(right_hand_descriptors, n_centroids)

            # print(type(left_hand_vector))
            # print(left_hand_vector.shape)

            left_hand_scaler = scaler.transform([left_hand_vector])
            left_hand_pca = pca.transform(left_hand_scaler)
            right_hand_scaler = scaler.transform([right_hand_vector])
            right_hand_pca = pca.transform(right_hand_scaler)

            left_result = pca_svm_model.predict(left_hand_pca)
            right_result = pca_svm_model.predict(right_hand_pca)
            # print('left: ', left_result)
            # print('left: ', right_result)


        # Thực hiện in kết quả:
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f'left: {left_result[0]}', (x1 + 30, y1 - 30), font, font_scale, font_color, font_thickness)
            cv2.putText(frame, f'right: {right_result[0]}', (x3 + 30, y3 - 30), font, font_scale, font_color, font_thickness)
            cv2.putText(frame, f'result:{left_result[0] * 10 + right_result[0]} ', (W - int(W/5), 30), font, font_scale, font_color, font_thickness)


            cv2.imshow('Image', frame)
            delay = int(1000 / fps) if fps > 0 else 1

            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break

    capture.release()
    cv2.destroyAllWindows()