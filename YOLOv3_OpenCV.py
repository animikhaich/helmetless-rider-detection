import os, sys, cv2, imutils, time, wget, shutil
import numpy as np
import itertools as it
import numpy as np
from libs.logger import logging

"""
To run this script, the weights, classes and configuration files for YOLOv3 will be required. 
Download links are as follows:
YOLOv3-320 Weights: https://pjreddie.com/media/files/yolov3.weights
YOLOv3-320 Config: https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
COCO Class Names: https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names

For a deeper YOLOv3 Architecture, Visit the Darkent homepage
Link: https://pjreddie.com/darknet/yolo/
"""


class ObjectTrackerYOLO:
    def __init__(self, yoloWeightsPath=None, minConfidence=0.4, minThreshold=0.4):

        self.urls = {
                'yolov3.weights': 'https://pjreddie.com/media/files/yolov3.weights',
                'yolov3.cfg': 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg',
                'coco.names': 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names'
            }

        # Set up YOLO weights path. If not given, download them
        if yoloWeightsPath:
            logging.info("Setting up predefined YOLO Weights Path: {}".format(yoloWeightsPath))
            self.yolo_path = yoloWeightsPath
        else:
            self.yolo_path = './weights/'
            logging.info("Setting YOLO Weights Path: {}".format(self.yolo_path))
        
        if not os.path.isdir(self.yolo_path):
            logging.info("Creating the YOLO Weights Directory: {}".format(self.yolo_path))
            os.mkdir(self.yolo_path)

        try:
            self.download_weights(self.urls)
        except Exception as e:
            logging.warning("YOLO weights download failed! Error Message: {}".format(e))
            logging.error("No Weights found, Exiting program...")
            exit(1)


        # Set Thresholds
        self.confidence = minConfidence
        self.threshold = minThreshold

        # derive the paths to the YOLO weights and model configuration
        self.weightsPath = os.path.join(self.yolo_path, "yolov3.weights")
        self.configPath = os.path.join(self.yolo_path, "yolov3.cfg")

        # Initializing the YOLO Network
        logging.info("Loading YOLO Network from Disk")
        self.net = cv2.dnn.readNetFromDarknet(self.configPath, self.weightsPath)
        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def download_weights(self, url_dict):
        """
        Download the YOLOv3 Weights
        """
        logging.info("Downloading Weights")
        for name in url_dict.keys():
            logging.info("Checking if {} exists".format(name))
            if os.path.exists(os.path.join(self.yolo_path, name)):
                logging.info("'{}' exists, hence skipping to the next file".format(name))
                continue
            logging.info("'{}' does not exist, hence Downloading File: {}".format(name, name))
            wget.download(url_dict[name], os.path.join(self.yolo_path, name))

               
    def per_frame_function(self, frame):
        """
        The main function which performs all the operations on each frame
        """
        H, W = frame.shape[:2]
        
        frame_copy = frame.copy()

        blob = cv2.dnn.blobFromImage(frame_copy, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        layerOutputs = self.net.forward(self.ln)
        

        # initialize our lists of detected bounding boxes, confidences,
        # and class IDs, respectively
        boxes = []
        confidences = []
        rects=[]

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections

            for detection in output:
                # extract the class ID and confidence (i.e., probability)
                # of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if classID != 0:
                    continue
                
                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > self.confidence:
                    box = detection[0:4] * np.array([W, H, W, H])

                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top
                    # and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates,
                    # confidences, and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
        
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, self.threshold)

        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                
                cropped_img = frame[y:y+h, x:x+w]
                
                rects.append([np.asarray([x, y, x+w, y+h]).astype("int"), cropped_img])
                cv2.rectangle(frame_copy, (x, y), (x+w, y+h), (255, 178, 50), 2)

        return frame_copy


if __name__ == '__main__':
    
    path = 'data/source/DSC_0001.MOV'
    cap = cv2.VideoCapture(path)

    yolo_object = ObjectTrackerYOLO(minThreshold=0.5, minConfidence=0.45)
    
    # loop over frames from the video file stream
    while True:
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            break
    
        modified_frame = yolo_object.per_frame_function(frame)

        if modified_frame is None:
            cv2.imshow("Window", frame)
            continue

        cv2.putText(modified_frame, f"FPS: {1/(time.time() - start): .1f}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA) 
        cv2.imshow("Window", modified_frame)


        key = cv2.waitKey(1)
        if(key == ord('q')):
            break

    cv2.destroyAllWindows()
    cap.release()
