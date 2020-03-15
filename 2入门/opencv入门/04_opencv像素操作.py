import cv2
img = cv2.imread('image0.jpg',1)
(b,g,r) = img[100,100]
print(b,g,r)# bgr
#10 100 --- 110 100
for i in range(1,100)
    img[10+i,100] = (255,0,0)
cv2.imshow('image',img)
cv2.waitKey(0) #1000 ms