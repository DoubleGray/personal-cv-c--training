# 彩色 直方图均衡化
import cv2
import numpy as np

img = cv2.imread('image0.jpg', 1)
cv2.imshow('src', img)
b, g, r = cv2.split(img)  # 通道分解
# cv2.imshow("b", b)
# cv2.imshow("g", g)
# cv2.imshow("r", r)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
# cv2.imshow("bH", bH)
# cv2.imshow("gH", gH)
# cv2.imshow("rH", rH)
result = cv2.merge((bH, gH, rH))  # 通道合成
cv2.imshow('dst', result)
cv2.waitKey(0)
