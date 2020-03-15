import cv2 as cv
import numpy as np


def detect_in_video():
    capture = cv.VideoCapture(0)
    cv.namedWindow("Video", cv.WINDOW_AUTOSIZE)
    print("摄像头已打开")
    while True:
        ret, frame = capture.read()
        frame = cv.flip(frame, 1)
        detect_obj(frame)
        c = cv.waitKey(10)
        if c == 27:  # ESC
            break


def lion_train_detect():
    # 计算Hog特征
    posNum = 820
    negNum = 1931

    featureArray = np.zeros((posNum + negNum, featureNum), np.float32)
    labelArray = np.zeros((posNum + negNum, 1), np.int32)

    # svm 监督学习 样本 标签 svm -》image hog
    print("正在计算pos_Hog特征")
    for i in range(posNum):
        src = cv.imread("pos//" + str(i + 1) + ".jpg")
        # cv.imshow("src", src)
        hist = Hog.compute(src, blockMove)
        for j in range(featureNum):
            featureArray[i, j] = hist[j]
        labelArray[i, 0] = 1  # 以固定列的形式在每一行附加标签（正样本 label 1）
    print("正在计算neg_Hog特征")
    for i in range(negNum):
        src = cv.imread("neg//" + str(i + 1) + ".jpg")
        hist = Hog.compute(src, blockMove)
        for j in range(featureNum):
            featureArray[posNum + i, j] = hist[j]
        labelArray[posNum + i, 0] = -1  # （负样本 label -1）

    # 创建svm
    print("正在创建svm")
    svm = cv.ml.SVM_create()
    svm.setType(cv.ml.SVM_C_SVC)
    svm.setKernel(cv.ml.SVM_LINEAR)
    svm.setC(1)
    print("训练svm中")
    ret = svm.train(featureArray, cv.ml.ROW_SAMPLE, labelArray)
    alpha = np.zeros(1, np.float32)
    print("alpha:")
    print(alpha)

    # 获取阈值判决枚限 rho
    rho = svm.getDecisionFunction(0, alpha)
    print("rho:")
    print(rho)

    # 计算myDetect对象
    alphaArray = np.zeros((1, 1), np.float32)
    supportVArray = np.zeros((1, featureNum), np.float32)
    resultArray = np.zeros((1, featureNum), np.float32)
    alphaArray[0, 0] = alpha
    resultArray = -1 * alphaArray * supportVArray
    myDetect = np.zeros((featureNum + 1), np.float32)
    for i in range(featureNum):
        myDetect[i] = resultArray[0, i]
    myDetect[featureNum] = rho[0]

    # 设置Hog+svm的分类器实现svm判决
    Hog.setSVMDetector(myDetect)


def detect_obj(img):
    # 识别
    # print("正在识别...")
    # img = cv.imread("1.jpg")

    # padding(可选)
    # 在原图外围添加像素,适当的pad可以提高检测的准确率（可能pad后能检测到边角的目标？）
    # 常见的pad size 有(8, 8), (16, 16), (24, 24), (32, 32).
    # 通常scale在1.01-1.5
    res = Hog.detectMultiScale(img, 0, (16, 16), (8, 8), 1.5, 2)
    print(res)

    # 获取识别结果
    x = res[0][0][0]  # 坐标存储在最后一维
    y = res[0][0][1]
    w = res[0][0][2]
    h = res[0][0][3]

    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0))
    cv.imshow("Video", img)


# Hog变量
winSize = (64, 128)
blockSize = (16, 16)
blockMove = (8, 8)
cellSize = (4, 4)
binNum = 9
# 计算特征个数
featureNum = ((winSize[0] - blockSize[0]) / blockMove[0] + 1) \
             * ((winSize[1] - blockSize[1]) / blockMove[1] + 1) \
             * (blockSize[0] / cellSize[0]) * (blockSize[1] / cellSize[1]) * binNum
featureNum = int(featureNum)
# featureNum = ((64-16)/8+1)*((128-16)/8+1)*(8/4*8/4)*9 = 3780
# print(featureNum)

# 创建Hog对象
Hog = cv.HOGDescriptor(winSize, blockSize, blockMove, cellSize, binNum)
# 调用训练api
lion_train_detect()
print("正在打开摄像头...")
detect_in_video()
cv.waitKey(0)
