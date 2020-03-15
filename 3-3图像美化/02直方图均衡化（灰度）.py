# 灰度 直方图均衡化
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('image0.jpg', 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hist = cv2.calcHist([gray], [0], None, [255], [0, 255])
plt.plot(hist)
plt.show()
cv2.imshow('src', gray)

dst = cv2.equalizeHist(gray)
hist2 = cv2.calcHist([dst], [0], None, [255], [0, 255])
plt.plot(hist2)
plt.show()
cv2.imshow('dst', dst)
cv2.waitKey(0)
