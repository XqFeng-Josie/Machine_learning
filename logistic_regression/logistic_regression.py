#coding=utf-8
#logistic回归的梯度上升法

from numpy import *
import matplotlib.pyplot as plt
#加载数据集
def loadDataSet():
    dataMat = [];
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])#x0=1
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inx):
    return 1.0 / (1 + exp(-inx));

#梯度上升，主要是采用了最大似然的推导
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMatrix)#n=3
    alpha = 0.001#学习率
    maxCycles = 500#循环轮数
    theta = ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * theta)
        error = (labelMat - h)
        theta = theta + alpha * dataMatrix.transpose() * error
    return theta
#根据训练好的theta绘图
def plotBestFit(theta):
    dataMat,labelMat = loadDataSet()
    dataArr =array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    #将数据按真实标签进行分类
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(xcord1, ycord1, s = 30, c = 'red', marker='s')
    ax.scatter(xcord2, ycord2, s = 30, c = 'blue')
    #生成x的取值 -3.0——3.0,增量为0.1
    x = arange(-3.0, 3.0, 0.1)
    #y = Θ0+Θ1x1+Θ2x2
    #y=x2
    y = (-theta[0] - theta[1] * x) / theta[2]
    ax.plot(x, y.T)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
dataMat,labelMat = loadDataSet()#加载数据集
theta = gradAscent(dataMat,labelMat)#计算参数
plotBestFit(theta)#根据参数画出分界线以及相应分类点
