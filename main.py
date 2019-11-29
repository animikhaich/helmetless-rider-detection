# Essentials
import os, wget, sys, shutil, imutils
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
    def __init__(self, yolo_weights='weights/yolo.h5', yolo_labels='weights/coco.names'):
        # Taking care of the YOLOv3 Weights & labels
        self.weights = self.verify_weghts(yolo_weights) # Returns Path of the weights --> String
        self.labels = self.verify_labels(yolo_labels)   # Retuns Dictionary

        # Initialize the constants
        self.yolo_anchors = np.array(
            [[10, 13],
            [16, 30],
            [33, 23],
            [30, 61],
            [62, 45],
            [59, 119],
            [116, 90],
            [156, 198],
            [373, 326]]
        )

        # Initializing the model architecture and loading weights
        self.init_model()

    def letterbox_image_custom(self, image, size):
        h, w = image.shape[:2]
        target_width, target_height = size
        
        if h < w:
            image = imutils.resize(image, width=target_width)
            h, w = image.shape[:2]
            diff = abs(h-target_height)//2
            top, bottom, left, right = (diff, diff, 0, 0)
        else:
            image = imutils.resize(image, height=target_height)
            h, w = image.shape[:2]
            diff = abs(target_width-w)//2
            top, bottom, left, right = (0, 0, diff, diff)
        
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, None, 0)
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        return image
        
    def init_model(self):
        """
        Function to Initialize the YOLOv3 model architecture and the weights.
        """
        self.model = self.yolo_main(Input(shape=(None, None, 3)), len(self.yolo_anchors)//3, len(self.labels))
        self.model.load_weights(self.weights)
        

    def yolo_main(self, input, num_anchors, num_classes):

        darknet_network = Model(input, self.darknet(input))

        network, network_1 = self.last_layers(darknet_network.output, 512, num_anchors * (num_classes + 5), layer_name="last1")

        network = self.NetworkConv2D_BN_Leaky( input=network, channels=256, kernel_size=(1,1))
        network = UpSampling2D(2)(network)
        network = Concatenate()([network, darknet_network.layers[152].output])

        network, network_2 = self.last_layers(network,  256,  num_anchors * (num_classes + 5), layer_name="last2")

        network = self.NetworkConv2D_BN_Leaky(input=network, channels=128, kernel_size=(1, 1))
        network = UpSampling2D(2)(network)
        network = Concatenate()([network, darknet_network.layers[92].output])

        network, network_3 = self.last_layers(network, 128, num_anchors * (num_classes + 5), layer_name="last3")

        return Model(input, [network_1, network_2, network_3])
    
    def darknet(self, input):
        network = self.NetworkConv2D_BN_Leaky(input=input, channels=32, kernel_size=(3,3))
        network = self.residual_block(input=network, channels=64, num_blocks=1)
        network = self.residual_block(input=network, channels=128, num_blocks=2)
        network = self.residual_block(input=network, channels=256, num_blocks=8)
        network = self.residual_block(input=network, channels=512, num_blocks=8)
        network = self.residual_block(input=network, channels=1024, num_blocks=4)
        
        return network


    def last_layers(self, input, channels_in, channels_out, layer_name=""):

        network = self.NetworkConv2D_BN_Leaky( input=input, channels=channels_in, kernel_size=(1,1))
        network = self.NetworkConv2D_BN_Leaky(input=network, channels= (channels_in * 2) , kernel_size=(3, 3))
        network = self.NetworkConv2D_BN_Leaky(input=network, channels=channels_in, kernel_size=(1, 1))
        network = self.NetworkConv2D_BN_Leaky(input=network, channels=(channels_in * 2), kernel_size=(3, 3))
        network = self.NetworkConv2D_BN_Leaky(input=network, channels=channels_in, kernel_size=(1, 1))

        network_1 = self.NetworkConv2D_BN_Leaky(input=network, channels=(channels_in * 2), kernel_size=(3, 3))
        network_1 = Conv2D(filters=channels_out, kernel_size=(1,1), name=layer_name)(network_1)

        return network, network_1


    def NetworkConv2D_BN_Leaky(self, input, channels, kernel_size, kernel_regularizer = l2(5e-4),
                            strides=(1,1), padding="same", use_bias=False):

        network = Conv2D( filters=channels, kernel_size=kernel_size, strides=strides, padding=padding,
                        kernel_regularizer=kernel_regularizer, use_bias=use_bias)(input)
        network = BatchNormalization()(network)
        network = LeakyReLU(alpha=0.1)(network)
        return network


    def residual_block(self, input, channels, num_blocks):
        network = ZeroPadding2D(((1,0), (1,0)))(input)
        network = self.NetworkConv2D_BN_Leaky(input=network,channels=channels, kernel_size=(3,3), strides=(2,2), padding="valid")

        for _ in range(num_blocks):
            network_1 = self.NetworkConv2D_BN_Leaky(input=network, channels= channels // 2, kernel_size=(1,1))
            network_1 = self.NetworkConv2D_BN_Leaky(input=network_1,channels= channels, kernel_size=(3,3))

            network = Add()([network, network_1])
        return network


    def verify_weghts(self, weight_path):
        logging.info("Checking if YOLOv3 pre-trained weights are present")
        try:
            if not (os.path.exists(weight_path) and os.stat(weight_path).st_size / 1E6 > 248):
                logging.info("Weights do not exist or are broken. Downloading Weights...")
                URL = "https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo.h5"

                if not os.path.isdir(os.path.split(weight_path)[0]): os.makedirs(os.path.split(weight_path)[0])

                wget.download(URL, out=weight_path)
                logging.info(f"YOLOv3 Weights Downloaded at: {weight_path}")
            else:
                logging.info("YOLOv3 pre-trained weights are already present.")
        except Exception as e:
            logging.error(f"Error Downloading Weights: {e}")
            exit(1)
        
        return weight_path


    def verify_labels(self, labels_path):
        logging.info("Checking if COCO labels are present")
        try:
            if not (os.path.exists(labels_path) and os.stat(labels_path).st_size > 600):
                logging.info("Labels do not exist or are broken. Downloading labels...")
                URL = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"

                if not os.path.isdir(os.path.split(labels_path)[0]): os.makedirs(os.path.split(labels_path)[0])

                wget.download(URL, out=labels_path)
                logging.info(f"COCO Labels Downloaded at: {labels_path}")
            else:
                logging.info("COCO labels are already present.")
        except Exception as e:
            logging.error(f"Error Downloading Labels: {e}")
            exit(1)
        
        with open(labels_path, 'r') as f:
            contents = f.read().strip()
        
        labels = {i: contents.split('\n')[i] for i in range(len(contents.split('\n')))}

        return labels



# Testing Code
if __name__ == '__main__':
    obj = YOLO()
        