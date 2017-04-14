#coding=utf-8
#测试局部加权线性回归算法
import matplotlib.pyplot as plt
import LocallyWeightedLinearRegression as lwlr
from numpy import *

xArr, yArr = lwlr.load_data('ex2.txt')

yHat = lwlr.lwlrTest(xArr, xArr, yArr, 1.0)
xMat = mat(xArr)
strInd = xMat[:, 1].argsort(0)#返回数据第二列从小到大的索引值
#xSort = xMat[strInd][:, 0, :]#返回第一列和第二列，根据上面索引顺序取出
xSort = xMat[strInd][:]#返回第一列和第二列，根据上面索引顺序取出,xSort为Xmat的另一个拷贝

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(xSort[:, 1], yHat[strInd])#画出拟合的线
ax.scatter(xMat[:, 1].flatten().A[0], mat(yArr).T.flatten().A[0], s = 2, c = 'red')#画出实际点
plt.show()