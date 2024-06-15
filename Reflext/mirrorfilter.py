import cv2
import numpy as np

image = cv2.imread('test1.jpg')

mirror_img = image.copy()

cv2.imshow('test1', mirror_img)
cv2.waitKey(0)
cv2.destroyAllWindows()