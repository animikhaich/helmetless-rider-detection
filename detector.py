import cv2

img = 'data/source/frame96.jpg'
img = cv2.imread(img, 0)
cv2.imshow("Test Img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()