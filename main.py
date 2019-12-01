# Essentials
import os, wget, sys, shutil, imutils, time
import numpy as np

# Computer Vision
import cv2

# Tensorflow
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, LeakyReLU, BatchNormalization, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model

# Custom Libs
from libs.logger import logging


class YOLO:
    def __init__(self, yolo_weights='weights/yolo.h5', yolo_labels='weights/coco.names', target_dims=(416, 416)):
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

        self.target_dims = target_dims

        # Initializing the model architecture and loading weights
        self.init_model()
        
        ### TEMP SECTION ###
        frame = cv2.imread('sample.jpg')
        preds = self.evaluate(frame)

        out_boxes, out_scores, out_classes = self.yolo_eval(preds, self.yolo_anchors, len(self.labels), self.target_dims)
        
        min_probability = 0.9        
        for a, b in reversed(list(enumerate(out_classes))):
            predicted_class = self.labels[np.int(b)]
            box = out_boxes[a]
            score = out_scores[a]

            if score < min_probability:
                continue

            label = "{} {:.2f}".format(predicted_class, score)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype(np.int32))
            left = max(0, np.floor(left + 0.5).astype(np.int32))
            bottom = min(frame.shape[1], np.floor(bottom + 0.5).astype(np.int32))
            right = min(frame.shape[0], np.floor(right + 0.5).astype(np.int32))

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 1)
            cv2.putText(frame, label, (left, top-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA) 
        
        cv2.imshow("Detections", frame)
        cv2.waitKey(0)

    def yolo_eval(self, yolo_outputs, anchors, num_classes, image_shape, max_boxes=20,score_threshold=.6, iou_threshold=.5):

        num_layers = len(yolo_outputs)
        anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]
        input_shape = tf.shape(yolo_outputs[0])[1:3] * 32
        boxes = []
        box_scores = []

        for l in range(num_layers):
            _boxes, _box_scores = self.yolo_boxes_and_scores(yolo_outputs[l],
                anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
            boxes.append(_boxes)
            box_scores.append(_box_scores)
        boxes = tf.concat(boxes, axis=0)
        box_scores = tf.concat(box_scores, axis=0)

        mask = box_scores >= score_threshold
        max_boxes_tensor = tf.constant(max_boxes, dtype='int32')
        boxes_ = []
        scores_ = []
        classes_ = []
        for c in range(num_classes):
            class_boxes = tf.boolean_mask(boxes, mask[:, c])
            class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
            nms_index = tf.image.non_max_suppression(
                class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
            class_boxes = tf.gather(class_boxes, nms_index)
            class_box_scores = tf.gather(class_box_scores, nms_index)
            classes = tf.ones_like(class_box_scores, 'int32') * c
            boxes_.append(class_boxes)

            scores_.append(class_box_scores)
            classes_.append(classes)
        boxes_ = tf.concat(boxes_, axis=0)
        scores_ = tf.concat(scores_, axis=0)
        classes_ = tf.concat(classes_, axis=0)

        return boxes_, scores_, classes_

    def yolo_head(self, feats, anchors, num_classes, input_shape, calc_loss=False):

        num_anchors = len(anchors)

        anchors_tensor = tf.reshape(tf.constant(anchors, dtype=tf.float32), [1, 1, 1, num_anchors, 2])

        grid_shape = tf.shape(feats)[1:3]
        grid_y = tf.tile(tf.reshape(tf.range(0, limit=grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], 1, 1])
        grid_x = tf.tile(tf.reshape(tf.range(0, limit=grid_shape[1]), [1, -1, 1, 1]), [grid_shape[0], 1, 1, 1])
        grid = tf.concat([grid_x, grid_y], axis=-1)
        grid = tf.cast(grid, feats.dtype)

        feats = tf.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

        box_xy = (tf.sigmoid(feats[..., :2]) + grid) / tf.cast(grid_shape[::-1], feats.dtype)
        box_wh = tf.exp(feats[..., 2:4]) * anchors_tensor / tf.cast(input_shape[::-1], feats.dtype)
        box_confidence = tf.sigmoid(feats[..., 4:5])
        box_class_probs = tf.sigmoid(feats[..., 5:])

        if calc_loss == True:
            return grid, feats, box_xy, box_wh
        return box_xy, box_wh, box_confidence, box_class_probs

    def yolo_correct_boxes(self, box_xy, box_wh, input_shape, image_shape):

        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = tf.cast(input_shape, box_yx.dtype)
        image_shape = tf.cast(image_shape, box_yx.dtype)
        new_shape = tf.round(image_shape * tf.reduce_min(input_shape/image_shape))
        offset = (input_shape-new_shape)/2./input_shape
        scale = input_shape/new_shape
        box_yx = (box_yx - offset) * scale
        box_hw *= scale

        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes =  tf.concat([
            box_mins[..., 0:1],
            box_mins[..., 1:2],
            box_maxes[..., 0:1],
            box_maxes[..., 1:2]
        ], axis=-1)


        boxes *= tf.concat([image_shape, image_shape], axis=-1)
        return boxes

    def yolo_boxes_and_scores(self, feats, anchors, num_classes, input_shape, image_shape):

        box_xy, box_wh, box_confidence, box_class_probs = self.yolo_head(feats, anchors, num_classes, input_shape)
        boxes = self.yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
        boxes = tf.reshape(boxes, [-1, 4])
        box_scores = box_confidence * box_class_probs
        box_scores = tf.reshape(box_scores, [-1, num_classes])
        return boxes, box_scores

    def evaluate(self, frame):
        frame = self.letterbox_image(frame, self.target_dims)
        frame = self.preprocess_image(frame)
        
        # Perform inference
        preds = self.model.predict(frame)

        return preds

    def preprocess_image(self, image):
        image = image / 255
        expanded_image = np.expand_dims(image, axis=0)
        return expanded_image

    def letterbox_image(self, image, size):
        h, w = image.shape[:2]
        tw, th = size
        
        if w/h > tw/th:
            image = imutils.resize(image, width=tw)
            h, w = image.shape[:2]
            diff = abs(th-h)//2
            top, bottom, left, right = (diff, diff, 0, 0)
        else:
            image = imutils.resize(image, height=th)
            h, w = image.shape[:2]
            diff = abs(tw-w)//2
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
        