import cv2 
# 1 文件的读取 2 封装格式解析 3 数据解码 4 数据加载
img = cv2.imread('image0.jpg',1)
cv2.imshow('image',img)
# jpg png  1 文件头 2 文件数据
cv2.imwrite('image1.jpg',img) # 1 name 2 data 
cv2.waitKey (0)
# 1.14M 130k