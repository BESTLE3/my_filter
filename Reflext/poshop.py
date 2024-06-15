import cv2
import numpy as np

alpha = 1.0

img = cv2.imread("F:/Dev/Reflext/Le_sserafim.jpg")
blurimg = cv2.bilateralFilter(img, 5, 75, 75)


outputimg = np.clip((1+alpha) * blurimg - 128 * alpha, 0, 255).astype(np.uint8)

cv2.imshow('img', img)
cv2.imshow('img2', outputimg)

cv2.waitKey(0)
cv2.destroyAllWindows()