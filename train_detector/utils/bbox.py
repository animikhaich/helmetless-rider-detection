import numpy as np
import os
import cv2
from .colors import get_color

class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, c = None, classes = None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        
        self.c       = c
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        
        return self.label
    
    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
            
        return self.score      

def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
             return 0
        else:
            return min(x2,x4) - x3    

def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])  
    
    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
    w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
    
    union = w1*h1 + w2*h2 - intersect
    
    return float(intersect) / union

def draw_boxes(image, boxes, labels, obj_thresh, quiet=True):
    image_copy = image.copy()
    final_boxes, final_labels, = [], []

    for box in boxes:
        label_str = ''
        label = -1
        
        for i in range(len(labels)):
            if box.classes[i] > obj_thresh:
                if label_str != '': label_str += ', '
                label_str += (labels[i] + ' ' + str(round(box.get_score()*100, 2)) + '%')
                label = i
            if not quiet: print(label_str)
                
        if label >= 0:
            text_size = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 1.1e-3 * image.shape[0], 5)
            width, height = text_size[0][0], text_size[0][1]
            region = np.array([[box.xmin-1,        box.ymin], 
                               [box.xmin-1,        box.ymin-height-10], 
                               [box.xmin+width-10, box.ymin-height-10], 
                               [box.xmin+width-10, box.ymin]], dtype='int32')  

            final_boxes.append([box.xmin, box.ymin, box.xmax, box.ymax])
            final_labels.append(label_str)

            cv2.rectangle(img=image_copy, pt1=(box.xmin,box.ymin), pt2=(box.xmax,box.ymax), color=get_color(label), thickness=2)
            cv2.fillPoly(img=image_copy, pts=[region], color=get_color(label))
            cv2.putText(img=image_copy, 
                        text=label_str, 
                        org=(box.xmin+3, box.ymin-7), 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale=1e-3 * image_copy.shape[0], 
                        color=(0,0,0), 
                        thickness=2)
        
    return image_copy, final_boxes, final_labels