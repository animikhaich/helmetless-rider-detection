import cv2


SOURCE_IMAGE = '/media/Data/Documents/Python-Codes/helmetless-rider-detection/data/selected_plates/DSC_0001_54_9_7.jpg'

image = cv2.imread(SOURCE_IMAGE)
cv2.imshow("Image", image)
cv2.waitKey(0)


MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15



cv2.destroyAllWindows()