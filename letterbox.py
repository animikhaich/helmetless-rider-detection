import cv2, imutils, time
import numpy as np
from PIL import Image

def letterbox_image_custom(image, size):
    h, w = image.shape[:2]
    tw, th = size
    
    if w/h > tw/th:
        image = imutils.resize(image, width=tw)
        h, w = image.shape[:2]
        diff = abs(h-th)//2
        top, bottom, left, right = (diff, diff, 0, 0)
    else:
        image = imutils.resize(image, height=th)
        h, w = image.shape[:2]
        diff = abs(tw-w)//2
        top, bottom, left, right = (0, 0, diff, diff)
    
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, None, 0)
    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return image


image = cv2.imread('sample.jpg')

img = letterbox_image_custom(image, (image.shape[0], image.shape[0]))
print(img.shape)

cv2.imshow("Sample 1", img)

cv2.waitKey(0)
cv2.destroyAllWindows()