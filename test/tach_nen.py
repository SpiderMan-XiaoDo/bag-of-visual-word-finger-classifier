import numpy as np
import cv2

if __name__ == '__main__':
    videopath = r'baocao_computervision\test\video_test\video_test.mp4'
    capture = cv2.VideoCapture(videopath)

    # Get the frame rate of the video
    fps = capture.get(cv2.CAP_PROP_FPS)
    index = 0
    while capture.isOpened():
        ret, frame = capture.read()

        if not ret:
            print("Error reading the video feed.")
            break


        converted2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        converted =  cv2.cvtColor(frame , cv2.COLOR_BGR2YCR_CB)
        min_ycc = np.array([0,140,85], np.uint8)
        max_ycc = np.array([255,176,140], np.uint8 )
        skinMask = cv2.inRange(converted, min_ycc, max_ycc)
        skinMask = cv2.addWeighted(skinMask,0.5,skinMask,0.5,0.0)             
        skinMask = cv2.medianBlur(skinMask, 5)

        
        skin = cv2.bitwise_and(converted2, converted2, mask = skinMask)
        cv2.imshow('Image', frame)
        index+=1
        # cv2.imwrite(f'test_bounder_{index}.jpg', skin)
        # Calculate the delay based on the original frame rate
        delay = int(1000 / fps) if fps > 0 else 1

        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()