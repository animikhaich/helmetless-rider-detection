import warnings
warnings.simplefilter('ignore')
from imageai.Detection import ObjectDetection
import os
import math
from YOLOv3 import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import cv2, os, time, logging

logging.basicConfig(filename="HeadExtractor.log", format='%(asctime)s %(message)s', filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def intersection(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return x1, y1, x2, y2


image_folder = "D:/Divided dataset/Animikh/"
yolo_weights_file = "C:/Users/animi/Desktop/Object Detection Using ImageAI/yolo.h5"
destination_folder = "D:/Divided dataset/Animikh_Heads/"
destination_folder_detected = "D:/Divided dataset/Animikh_Detected/"

test_obj = YOLO(yolo_weights_file)

start_time = time.time()
test_obj.init_model()
print("Checkpoint 1...", time.time() - start_time)

images = sorted([file for file in os.listdir(image_folder) if '.jpg' in file])

for iteration, image in enumerate(images):
    start_time = time.time()
    frame_original = cv2.imread(image_folder+image)
    frame = frame_original.copy()
    detections = test_obj.evaluate_frame(frame)
    print("START Checkpoint for Iteration: {}, Image {} Time taken for detection: {}".format(iteration+1, image, time.time()-start_time))
    logger.info("START Checkpoint for Iteration: {}, Image {} Time taken for detection: {}".format(iteration+1, image, time.time()-start_time))

    head_detected = frame_original.copy()
    detected_copy = frame_original.copy()

    try:
        start_time = time.time()
        list_of_rects = []
        list_of_heads = []
        for eachObject in detections:
            if eachObject['name'] == 'motorcycle':
                for secondObject in detections:
                    if secondObject['name'] == 'person' and abs(eachObject['box_points'][0]-secondObject['box_points'][0]) < 50:
                        intersection_rect = intersection(eachObject['box_points'], secondObject['box_points'])
                        intersection_top_left = (intersection_rect[0], intersection_rect[1])
                        intersection_bottom_right = (intersection_rect[2], intersection_rect[3])
                        intersection_width = intersection_rect[2] - intersection_rect[0]
                        intersection_height = intersection_rect[3] - intersection_rect[1]
                        area = intersection_width * intersection_height

                        bigbox_left = min(secondObject['box_points'][0], eachObject['box_points'][0])
                        bigbox_top = min(secondObject['box_points'][1], eachObject['box_points'][1])
                        bigbox_right = max(secondObject['box_points'][2], eachObject['box_points'][2])
                        bigbox_bottom = max(secondObject['box_points'][3], eachObject['box_points'][3])

                        person_height = secondObject['box_points'][3] - secondObject['box_points'][1]
                        person_left_top = (secondObject['box_points'][0], secondObject['box_points'][1])
                        person_right_bottom = (secondObject['box_points'][2], secondObject['box_points'][3] - 3 * person_height //4)

                        if area > 0 and area > 8000:
                            list_of_rects.append((bigbox_left, bigbox_top, bigbox_right, bigbox_bottom))
                            cv2.rectangle(detected_copy, (bigbox_left, bigbox_top), (bigbox_right, bigbox_bottom), (0, 255, 0),  2)

                            list_of_heads.append((person_left_top[0], person_left_top[1], person_right_bottom[0], person_right_bottom[1]))
                            # cv2.rectangle(head_detected, person_left_top, person_right_bottom, (0, 255, 0),  2)

        cv2.imwrite(destination_folder_detected + image.split('.')[0] + str(iteration) + '.jpg', detected_copy)
        im = frame_original.copy()
        j = 1
        num_heads = len(list_of_heads)
        for num, i in enumerate(list_of_heads):
            cropped_img = im[i[1]:i[3], i[0]:i[2]]
            print("###### Writing {} of {} Heads to Destination folder".format(num+1, num_heads))
            cv2.imwrite(destination_folder + image.split('.')[0] + str(j) + '.jpg', cropped_img)
            j += 1

        print("END Checkpoint for Iteration: {} || Image {} || Detections: {} || Time taken for Cropping: {}".format(iteration + 1, image,
                                                                        num_heads, time.time() - start_time))
        logger.info(
            "END Checkpoint for Iteration: {} || Image {} || Detections: {} || Time taken for Cropping: {}".format(iteration + 1, image,
                                                                        num_heads, time.time() - start_time))

    except Exception as e:
        print("||*****ERROR*****|| at iteration:", iteration + 1, "Image:", image)
        logging.info("||*****ERROR*****|| at iteration:", iteration + 1, "Image:", image)
        continue