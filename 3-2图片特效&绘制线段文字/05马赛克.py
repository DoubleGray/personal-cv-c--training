import cv2
import numpy as np

img = cv2.imread('image0.jpg', 1)
imgInfo = img.shape
height = imgInfo[0]
width = imgInfo[1]
# unit_x和unit_y表马赛克单元尺寸
unit_x = 10
unit_y = 10
mosaic_unit_size = [unit_y, unit_x]
for m in range(100, 300):
    for n in range(100, 300):
        # pixel ->10*10
        if m % unit_y == 0 and n % unit_x == 0:
            for i in range(0, unit_y):
                for j in range(0, unit_x):
                    (b, g, r) = img[m, n]
                    img[i + m, j + n] = (b, g, r)
cv2.imshow('dst', img)
cv2.waitKey(0)
