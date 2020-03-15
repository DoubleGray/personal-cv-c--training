import cv2
import numpy as np

img = cv2.imread('image0.jpg', 1)
for i in range(200, 300):
    img[i, 200] = (255, 255, 255)
    img[i, 200 + 1] = (255, 255, 255)
    img[i, 200 - 1] = (255, 255, 255)
for i in range(150, 250):
    img[250, i] = (255, 255, 255)
    img[250 + 1, i] = (255, 255, 255)
    img[250 - 1, i] = (255, 255, 255)
# cv2.imwrite('damaged.jpg', img)
cv2.imshow('image', img)
cv2.waitKey(0)
