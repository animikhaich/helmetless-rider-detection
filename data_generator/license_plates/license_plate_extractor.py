import os, sys, cv2
from tqdm import tqdm
from glob import glob

sys.path.append("")
from train_detector.object_detector import YOLOObjectDetector
from libs.centroidtracker import CentroidTracker
from libs.logger import logging
from libs.opencv_utils import *

SOURCE_VIDEO_DIR = 'data/source'
DEST_PLATES_DIR = 'data/dest/captured_frames'

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

# Check for the source and destination directories and create appropriate folders
if not os.path.isdir(DEST_PLATES_DIR): os.makedirs(DEST_PLATES_DIR)

# Read all the videos present in the folder
filepaths = glob(os.path.join(SOURCE_VIDEO_DIR, "*"))

# Create Instance of YOLOObjectDetector
detector = YOLOObjectDetector()

tracker = CentroidTracker(maxDisappeared=10)

for file_num, filepath in enumerate(filepaths):
    logging.info(f"Processing Video {file_num+1} out of {len(filepaths)} | File: {filepath}")
    
    # Initialize Video Capture
    cap = cv2.VideoCapture(filepath)

    # Video Meta
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    for frame_num in tqdm(range(total_frames)):
        _, frame = cap.read()

        if frame is None: 
            logging.warn(f"No frame found, continuing to the next one")
            continue
        
        try:
            modified_frame, final_boxes, final_labels = detector.detect(frame)
        except Exception as e:
            logging.error(f"Could not Detect Object in frame {frame_num+1} out of {total_frames} due to Error: {e}")
            continue
        
        if modified_frame is None:
            # terminate = show_feed(frame)
            # if terminate: break
            continue
        
        object_num = 0
        final_boxes = fix_boxes(final_boxes, frame_width, frame_height)
        
        # Discard boxes for other classes
        final_boxes = [box for box, label in zip(final_boxes, final_labels) if " ".join(label.split(' ')[:-1]).strip() == "License Plate"]

        objects = tracker.update(final_boxes)

        print(len(final_boxes), len(objects.items()))
        for box, label, (objectID, centroid) in zip(final_boxes, final_labels, objects.items()):
            object_num += 1
            cleaned_label = " ".join(label.split(' ')[:-1]).strip()

            # Skip for Person Bike
            if cleaned_label == "Person Bike": continue
            dir_path = os.path.join(DEST_PLATES_DIR, cleaned_label, str(objectID))

            if not os.path.isdir(dir_path): os.makedirs(dir_path)

            try:
                video_name = "".join(os.path.split(filepath)[-1].split('.')[:-1])
                roi_filepath = os.path.join(dir_path, f"{video_name}_{frame_num}_{objectID}_{object_num}.jpg")
            except Exception as e:
                logging.error(f"Unable to generate ROI Path for frame {frame_num+1} out of {total_frames} due to error: {e}")
                continue
            
            text = "ID {}".format(objectID)
            cv2.putText(modified_frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 3)
            cv2.circle(modified_frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
        
            try:
                x1, y1, x2, y2 = fix_box(box[0], box[1], box[2], box[3], frame_width, frame_height)

                object_image = frame[y1:y2, x1:x2]
                cv2.imwrite(roi_filepath, object_image)
            except Exception as e:
                logging.error(f"Unable to save ROI image for frame {frame_num+1} out of {total_frames} for Object_path: {roi_filepath} for Coordinates: {box} due to error: {e}")
            
        
        terminate = show_feed(modified_frame)
        if terminate: break


cap.release()
cv2.destroyAllWindows()
    
    