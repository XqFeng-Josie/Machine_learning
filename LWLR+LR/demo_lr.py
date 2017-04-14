#coding=utf-8
#测试一般线性回归算法
import matplotlib.pyplot as plt
import LocallyWeightedLinearRegression as lr
from numpy import *

xVec, yVec = lr.load_data('ex2.txt')
theta = lr.linearRegres(xVec, yVec)
xMat = mat(xVec)
yMat = mat(yVec)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0], s = 2, c = 'red')
xCopy = xMat.copy()
xCopy.sort(0)   # 把点按升序排列
yHat = xCopy * theta
ax.plot(xCopy[:,1],yHat)#画出拟合的线
plt.show()