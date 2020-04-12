from libs.object_detector import YOLOObjectDetector
from libs.logger import logging
# from keras.backend.tensorflow_backend import set_session
# from tensorflow.keras.models import Model
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import cv2
import time
import os
import imutils

# Variables
VIDEO_PATH = "data/raw_videos/DSC_0001.MOV"

# Initialise Object Detector
detector = YOLOObjectDetector(
    obj_thresh=0.85, nms_thresh=0.25, input_img_dims=800)

# Video Reader
cap = cv2.VideoCapture(VIDEO_PATH)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 60

for i in tqdm(range(total_frames)):
    start_time = time.time()
    _, frame = cap.read()

    modified_frame = frame
    if i > 500:
        modified_frame, boxes, labels = detector.detect(frame)

    # Display the FPS on the frame
    cv2.putText(
        modified_frame,
        f"FPS: {fps:.2f}",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )

    # Display Frame
    cv2.imshow("Stream", modified_frame)

    # Useless Waitkeys. Don't know why OpenCV uses this
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord(' '):
        cv2.waitKey(0)

    # Calculate the FPS
    fps = 1/(time.time() - start_time)

cap.release()
cv2.destroyAllWindows()
