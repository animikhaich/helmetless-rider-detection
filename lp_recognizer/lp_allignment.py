import numpy as np
import cv2


SOURCE_IMAGE = '/media/Data/Documents/Python-Codes/helmetless-rider-detection/data/selected_plates/DSC_0001_17302_669_4.jpg'

# Read Image in Grayscale
image = cv2.imread(SOURCE_IMAGE, 0)

# Different Simple Thresholding Techniques
ret, thresh1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Adaptive Thresholding
th2 = cv2.adaptiveThreshold(
    image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
th3 = cv2.adaptiveThreshold(
    image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Image Meta and showing
titles = ['Original Image', 'BINARY', 'Adaptive Mean Thresholding',
          'Adaptive Gaussian Thresholding']


images = [image, thresh1, th2, th3]

for title, image in zip(titles, images):
    cv2.imshow(title, image)


# img = cv2.imread(SOURCE_IMAGE)
# rgb_planes = cv2.split(img)

result_planes = []
result_norm_planes = []
for plane in rgb_planes:
    dilated_img = cv2.dilate(plane, np.ones((3, 3), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)
    diff_img = cv2.absdiff(plane, bg_img)
    norm_img = cv2.normalize(
        diff_img,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8UC1
    )
    result_planes.append(diff_img)
    result_norm_planes.append(norm_img)

# result = cv2.merge(result_planes)
# result_norm = cv2.merge(result_norm_planes)

# cv2.imshow('Original', img)
# cv2.imshow('bg_img', bg_img)
# cv2.imshow('diff_img', diff_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
