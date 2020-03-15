import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1 准备data
rand1 = np.array([[155, 48], [159, 50], [164, 53], [168, 56], [172, 60]])  # 女生
rand2 = np.array([[152, 53], [156, 55], [160, 56], [172, 64], [176, 65]])  # 男生

train_data = np.vstack((rand1, rand2))
train_data = np.array(train_data, np.float32)
print(train_data)

# 2 label

label = np.array([[0], [0], [0], [0], [0], [1], [1], [1], [1], [1]])
print(label)
# svm
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setC(0.1)

# train svm
ret = svm.train(train_data, cv2.ml.ROW_SAMPLE, label)

# predict
my_data = np.array([[178, 70], [168, 46]], np.float32)
print(my_data)
ret, result = svm.predict(my_data)

print(result[0])
print(result[1])
