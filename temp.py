#OBJECT DETECTION FROM IMAGE

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 16:28:45 2018

@author: AkhileshAkku
"""

from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()


detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath( os.path.join(execution_path , "yolo.h5"))
detector.loadModel()


custom_objects = detector.CustomObjects(person=True, motorcycle=True)
detections = detector.detectCustomObjectsFromImage(custom_objects=custom_objects, input_image=os.path.join(execution_path , "frame982.jpg"), output_image_path=os.path.join(execution_path , "frame982yolooutput.jpg"), minimum_percentage_probability=30)

print(detections)
# =============================================================================
# for eachObject in detections:
#     print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
#     print("--------------------------------")
# 
# =============================================================================
#OBJECT DETECTION FROM CUSTOM VIDEO

# =============================================================================
# from imageai.Detection import VideoObjectDetection
# import os
# 
# execution_path = os.getcwd()
# 
# detector = VideoObjectDetection()
# detector.setModelTypeAsRetinaNet()
# detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
# detector.loadModel()
# 
# custom_objects = detector.CustomObjects(person=True, motorcycle=True)
# 
# video_path = detector.detectCustomObjectsFromVideo(custom_objects=custom_objects, input_file_path=os.path.join(execution_path, "test_video2.mp4"),
#                                 output_file_path=os.path.join(execution_path, "traffic_custom_detected")
#                                 , frames_per_second=15, log_progress=True)
# print(video_path)
# ==========================================================================
    
import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import cv2

frame = cv2.imread("frame982yolooutput.jpg")
detected_copy = frame.copy()
detected_copy = cv2.cvtColor(detected_copy, cv2.COLOR_BGR2RGB)

head_detected = detected_copy.copy()

def intersection(a,b):
  x1 = max(a[0], b[0])
  y1 = max(a[1], b[1])
  x2 = min(a[2], b[2])
  y2 = min(a[3], b[3])
  return (x1, y1, x2, y2)   

list_of_rects = []
list_of_heads = []
for eachObject in detections:
    if eachObject['name'] == 'motorcycle':
        for secondObject in detections:
            if secondObject['name'] == 'person' and  abs(eachObject['box_points'][0]-secondObject['box_points'][0]) < 50:
                intersection_rect = intersection(eachObject['box_points'], secondObject['box_points'])
                intersection_top_left = (intersection_rect[0], intersection_rect[1])
                intersection_bottom_right = (intersection_rect[2], intersection_rect[3])
                intersection_width = intersection_rect[2] - intersection_rect[0]
                intersection_height = intersection_rect[3] - intersection_rect[1]
                area = intersection_width * intersection_height
                print(area)
                
                bigbox_left = min(secondObject['box_points'][0], eachObject['box_points'][0])
                bigbox_top = min(secondObject['box_points'][1], eachObject['box_points'][1])
                bigbox_right = max(secondObject['box_points'][2], eachObject['box_points'][2])
                bigbox_bottom = max(secondObject['box_points'][3], eachObject['box_points'][3])
                
                person_height = secondObject['box_points'][3] - secondObject['box_points'][1]
                person_left_top = (secondObject['box_points'][0], secondObject['box_points'][1])
                person_right_bottom = (secondObject['box_points'][2], secondObject['box_points'][3] - 3 * person_height //4)
# =============================================================================
#                 if area>0: 
#                     cv2.rectangle(detected_copy, intersection_top_left, intersection_bottom_right, (0,255,0),  2)
# =============================================================================
                
                if area > 0 and area > 8000 :
                    list_of_rects.append((bigbox_left, bigbox_top, bigbox_right, bigbox_bottom))
                    cv2.rectangle(detected_copy, (bigbox_left, bigbox_top), (bigbox_right, bigbox_bottom), (0,255,0),  2)
# cv2.line(detected_copy, top_left, bottom_right, (0, 255, 0), 2, lineType = 8)
                    cv2.imwrite("detected.jpg", detected_copy)
                    
                    
                    
                    list_of_heads.append((person_left_top[0], person_left_top[1], person_right_bottom[0], person_right_bottom[1]))
                    cv2.rectangle(head_detected, person_left_top, person_right_bottom, (0,255,0),  2)
                    cv2.imwrite("head_detected.jpg", head_detected)
                    

print(list_of_heads)
cv2.imshow("detected", detected_copy)

im = cv2.imread("head_detected.jpg")    
j = 1
for i in list_of_heads:
    cropped_img = im[i[1]:i[3], i[0]:i[2]]
    cv2.imwrite('croppedhead'+str(j)+'.jpg', cropped_img)
    j += 1
    
#cv2.imshow('cropped', cropped_img)
    



# =============================================================================
# im = np.array(Image.open('frame246yolooutput.jpg'), dtype=np.uint8)
# 
# # Create figure and axes
# fig,ax = plt.subplots(1)
# 
# # Display the image
# ax.imshow(im)
# 
# # Create a Rectangle patch
# rect = patches.Rectangle((intersection_rect[0], intersection_rect[1]), intersection_rect[2], intersection_rect[3],linewidth=2,edgecolor='g', facecolor = 'None')
# 
# # Add the patch to the Axes
# ax.add_patch(rect)
# 
# plt.show()    
# =============================================================================

    
    