from facenet_pytorch import MTCNN
import cv2
from PIL import Image as Img
from PIL import ImageTk
import numpy as np
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import os

# Mặc định threshold = [0.6, 0.7, 0.7]
mtcnn = MTCNN(margin=20, keep_all=True, post_process=False, device='cpu')

import glob
import torch
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
import os
from PIL import Image
import numpy as np
import joblib

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


def trans(img):
    transform = transforms.Compose([
            transforms.ToTensor(),
            fixed_image_standardization
        ])
    return transform(img)


def fixed_image_standardization(image_tensor):
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor


def save_knn_model(knn_model, file_path):
    print('saving knn_model!!!')
    joblib.dump(knn_model, f'{file_path}/knn_model.pkl', compress=3)
def save_features(data, filepath):
    print('saving feature!!!')  
    joblib.dump(data, filepath)
    print('saved')
def load_features(file_path):
    print('loading file')
    print('loaded')
    return joblib.load(file_path)

model = InceptionResnetV1(
    classify=False,
    pretrained="casia-webface"
).to(device)

model.eval()



knn_model = load_features(r'D:\WorkSpace\KhaiPhaDuLieu\classifier\dataset\knn_model\knn_model.pkl')

import streamlit as st
import cv2

st.title('Fingermath App')

#Khanh Duong OK
#Cong Son OK
# Mặc định threshold = [0.6, 0.7, 0.7]
# v_cap.release()
mtcnn = MTCNN(margin=20, keep_all=True, post_process=False, device='cpu')

import cv2
from IPython.display import display, Image as IPImage
video_url = r'D:\WorkSpace\KhaiPhaDuLieu\classifier\dataset\test_video\DatVan_v2.mp4'
v_cap = cv2.VideoCapture(video_url)

# v_cap = cv2.VideoCapture(0)


# v_cap  = cv2.VideoCapture(0)
v_cap .set(3, 1080)
v_cap .set(4, 720)
frame_placeholder = st.empty()

stop_button_pressed = st.button('Stop')



fps = v_cap.get(cv2.CAP_PROP_FPS)
index = 0
while v_cap .isOpened() and not stop_button_pressed:

    try:
        success, frame = v_cap.read()

        if not success:
            st.write("The video capture has ended.")
            break
        

        frame_copy = frame.copy()
        index +=1
        if not success:
            break
        file_name = r'D:\WorkSpace\KhaiPhaDuLieu\classifier\face_recognition\app\dataset\test_video\test.jpg'
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        mtcnn(frame, file_name)
        image = Image.open(file_name)
        # cv2.imshow('hello', image)
        embed = model(trans(image).unsqueeze(0).to(device))
        embed_numpy = embed.detach().cpu().numpy()
        label = knn_model.predict(embed_numpy)
        boxes, _ = mtcnn.detect(frame)
        # dict_labels 
        print(label)
        if index == 20:
            break
        label = str(label)
        for box in boxes:
            bbox = list(map(int,box.tolist()))
            print(bbox)
            frame_copy = cv2.rectangle(frame_copy, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,0,255), 6)
            frame_copy = cv2.putText(frame_copy, label , (bbox[0],bbox[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,0), 2, cv2.LINE_8)
        # cv2.imshow('resul1t', frame_copy)
        frame_copy  = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_copy, channels='RGB')
        delay = int(1000 / fps) if fps > 0 else 1
        if cv2.waitKey(1) & 0xFF == ord('q') or stop_button_pressed:
            break
    except Exception as e:
        frame_copy  = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_copy, channels='RGB')
        delay = int(1000 / fps) if fps > 0 else 1
        if cv2.waitKey(1) & 0xFF == ord('q') or stop_button_pressed:
            break
        print(label)
v_cap.release()
cv2.destroyAllWindows()
