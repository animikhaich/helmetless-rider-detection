import os
import sys
import argparse
import json
import cv2
from libs.logger import logging
from train_detector.utils.utils import get_yolo_boxes, makedirs
from keras.backend.tensorflow_backend import set_session
from train_detector.utils.bbox import draw_boxes
from keras.models import load_model
from tqdm import tqdm
import numpy as np
import tensorflow as tf

# TODO: [Complete Refactor] Convert all the functions into class format
# TODO: Add Support for Tensorflow 2.x

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


class YOLOObjectDetector:
    """
    Generalized YOLOv3 Detector implemented with keras for Tensorflow 1.x
    """

    def __init__(self, config_path='weights/config.json', input_img_dims=416, obj_thresh=0.85, nms_thresh=0.25):
        """
        __init__ Object Detector constructor

        This is the class constructor for the YOLOv3 Object Detector.
        It does the following functions:
        --> Tests if GPU is available, if available, it sets the "Allow memory growth" flag
        --> It reads the model hyperparameter from the config file
        --> It Initializes and loads them model into memory ready for inference

        Args:
            config_path (str, optional): The path for the Hyperparameter config JSON. Defaults to 'weights/config.json'.
            input_img_dims (int, optional): Input image dimensions (square image). Defaults to 416.
            obj_thresh (float, optional): The minimum confidence threshold for the object detection model. Defaults to 0.85.
            nms_thresh (float, optional): The minimum threshold to calculate the NMS boxes. Defaults to 0.25.
        """

        # Verify if GPU is available, if so then allow memory growth
        if tf.test.is_gpu_available():
            if int(tf.__version__.split('.')[0]) == 1:
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                session = tf.Session(config=config)
                set_session(session)
            else:
                gpu_devices = tf.config.experimental.list_physical_devices(
                    'GPU'
                )
                tf.config.experimental.set_memory_growth(gpu_devices[0], True)

        # Load Model Hyperparameter Config
        try:
            logging.info(
                f"Loading Model Hyperparameter Config form JSON: {config_path}")
            with open(config_path) as config_buffer:
                self.config = json.load(config_buffer)
        except Exception as e:
            logging.error(
                f"Unable to load Model Hyperparameter Config due to error: {e}")
            exit(1)

        # Initialize Network parameters
        self.net_h, self.net_w = input_img_dims, input_img_dims
        self.obj_thresh, self.nms_thresh = obj_thresh, nms_thresh

        # Load the model into memory
        os.environ['CUDA_VISIBLE_DEVICES'] = self.config['train']['gpus']
        logging.info(
            f"Environment Initialized for GPUs: {self.config['train']['gpus']}")
        try:
            logging.info(f"Loading Model")
            self.infer_model = load_model(
                self.config['train']['saved_weights_name'])
            logging.info(f"Model Loaded")
        except Exception as e:
            logging.error(f"Failed to Load Model due to Error: {e}")

    def detect(self, image):
        """
        detect objects in the frame/image passed as argument

        Args:
            image (numpy array): Pass the RGB/Grayscale image for object detection

        Returns:
            tuple: image with bounding boxes (numpy array), bounding box coordinates (list of list), labels (list of strings)
            tuple: returns None, None, None for incorrect arguments
        """

        # Verify if the image has 3 dimensions or not. if not, then add a pseudo dimension to prevent errors
        try:
            if len(image.shape) < 3:
                image = np.expand_dims(image, axis=2)
        except Exception as e:
            logging.error(
                f"Incorrect Argument Passed. Failed to detect due to error: {e}")
            return None, None, None

        # Run inference and get the bounding boxes
        batch_boxes = get_yolo_boxes(self.infer_model, [
                                     image], self.net_h, self.net_w, self.config['model']['anchors'], self.obj_thresh, self.nms_thresh)

        # Decode the output and get the corresponding labels and boxes
        image_copy, final_boxes, final_labels = draw_boxes(
            image, batch_boxes[0], self.config['model']['labels'], self.obj_thresh)

        return image_copy, final_boxes, final_labels
