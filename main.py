# Essentials
import os, wget, sys, shutil
import numpy as np

# Computer Vision
import cv2

# Tensorflow
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, MaxPool2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, LeakyReLU, BatchNormalization, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model

# Custom Libs
from libs.logger import logging


class YOLO:
    def __init__(self, yolo_weights='weights/yolo.h5'):
        self.weights = self.verify_weghts(yolo_weights)


    def verify_weghts(self, weight_path):
        logging.info("Checking if YOLOv3 pre-trained weights are present")
        try:
            if not (os.path.exists(weight_path) and os.stat(weight_path).st_size / 1E6 > 248):
                logging.info("Weights do not exist or are broken. Downloading Weights...")
                URL = "https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo.h5"

                if not os.path.isdir(os.path.split(weight_path)[0]): os.makedirs(os.path.split(weight_path)[0])

                wget.download(URL, out=weight_path)
                logging.info(f"YOLOv3 Weights Downloaded at: {weight_path}")
        except Exception as e:
            logging.error(f"Error Downloading Weights: {e}")
            exit(1)
        
        return weight_path



# Testing Code
if __name__ == '__main__':
    obj = YOLO()
        