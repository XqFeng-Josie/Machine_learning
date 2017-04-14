#coding=utf-8
#LocallyWeightedLinearRegression.py
#局部加权线性回归算法的实现
#其中线性回归使用的最小二乘法,因此回归系数是 theta = (X.T* W * X).I * X.T *W * Y
from numpy import *
#加载数据
def load_data(fileName):
    numFeat = len(open(fileName).readline().split(',')) - 1
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split(',')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr) #实际数据集
        labelMat.append(float(curLine[-1]))#实际数据标签值
    return dataMat,labelMat


# 对某一点计算估计值
def lwlr(testPoint, xArr, yArr, k = 1.0):
    xMat = mat(xArr); yMat = mat(yArr).T#矩阵化
    m = shape(xMat)[0]#取得维度25
    weights = mat(eye((m)))#eye生成对角矩阵,m*m
    for i in range(m):
        diffMat = testPoint - xMat[i, :]#计算测试点坐标和所有数据坐标的差值
        #计算权值 w =exp（(-（xi-X）^2)/(2*k*k)）
        weights[i, i] = exp(diffMat * diffMat.T/(-2.0 * k**2))      # 计算权重对角矩阵
    xTx = xMat.T * (weights * xMat)     #对x值进行加权计算          # 奇异矩阵不能求逆
    if linalg.det(xTx) == 0.0:
        print('This Matrix is singular, cannot do inverse')
        return
    #theta = (X.T* W * X).I * X.T *W * Y
    theta = xTx.I * (xMat.T * (weights * yMat))                     # 计算回归系数 ,对y加权
    return testPoint * theta
# 普通线性回归linear regression 计算回归系数
def linearRegres(xVec, yVec):
    xMat = mat(xVec);
    yMat = mat(yVec).T;
    xTx = xMat.T * xMat;
    if linalg.det(xTx) == 0:        # 奇异矩阵不能求逆
        print('This matrix is singular, cannot do inverse')
        return
    theta = xTx.I * xMat.T * yMat
    return theta
# 对所有点计算估计值
def lwlrTest(testArr, xArr, yArr, k = 1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)#初始化预测值列表
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat

