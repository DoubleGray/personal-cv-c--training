# YUV 直方图均衡化
import cv2
import numpy as np

img = cv2.imread('image0.jpg', 1)
cv2.imshow('src', img)
imgYUV = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
channelYUV = cv2.split(imgYUV)
# for i in range(len(channelYUV)):
#     cv2.imshow("channelYUV" + str(i), channelYUV[i])
channelYUV[0] = cv2.equalizeHist(channelYUV[0])
channels = cv2.merge(channelYUV)
result = cv2.cvtColor(channels, cv2.COLOR_YCrCb2BGR)
cv2.imshow('dst', result)
cv2.waitKey(0)
