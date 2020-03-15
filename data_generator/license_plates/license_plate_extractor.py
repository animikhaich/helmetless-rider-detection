import os, sys, cv2
from tqdm import tqdm
from glob import glob

sys.path.append("")
from train_detector.object_detector import YOLOObjectDetector
from libs.logger import logging
from libs.opencv_utils import *

SOURCE_VIDEO_DIR = 'data/source'
DEST_PLATES_DIR = 'data/dest/plates'


# Check for the source and destination directories and create appropriate folders
if not os.path.isdir(DEST_PLATES_DIR): os.makedirs(DEST_PLATES_DIR)

# Read all the videos present in the folder
filepaths = glob(os.path.join(SOURCE_VIDEO_DIR, "*"))

# Create Instance of YOLOObjectDetector
detector = YOLOObjectDetector()

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
        
        # try:
        modified_frame, final_boxes, final_labels = detector.detect(frame)
        # except Exception as e:
        #     logging.error(f"Could not Detect Object in frame {frame_num+1} out of {total_frames} due to Error: {e}")
        #     continue
        
        if modified_frame is None:
            terminate = show_feed(frame)
            if terminate: break
            continue
            
        terminate = show_feed(modified_frame)
        if terminate: break


cap.release()
cv2.destroyAllWindows()
    
    