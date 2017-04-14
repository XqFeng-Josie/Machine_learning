#coding=utf-8
#梯度下降算法的实现
import numpy as np
import matplotlib.pyplot as plt
#学习率设置
rate = 0.001
x_train = np.array([[1, 2],[2, 1],[2, 3],[3, 5],[1, 3],[4, 2],[7, 3],[4, 5],[11, 3],[8, 7]])
y_train = np.array([7, 8, 10, 14, 8, 13, 20, 16, 28, 26])
x_test  = np.array([[1, 4],[2, 2],[2, 5],[5, 3],[1, 5],[4, 1]])

m = np.shape(x_train)[1]
#初始化参数
theta = np.random.normal(size=m)
#定义计算预测值的函数 h(x) = theta * x
def h(x):
    return np.dot(theta,x.T)
#梯度下降的规则是：
#J对theta的导数是： (h(x)-y)*x
#theta = theta - alpha * (h(x)-y)*x
#设置了两个迭代条件：
# 循环次数达到一定数目
# theta更新差低于某个数
for i in range(100):
    theta_old = np.zeros(m)
    for x, y in zip(x_train, y_train):
        theta_old = theta_old +rate *(y-h(x))* x
    if(np.dot((theta_old),(theta_old).T)<0.001):
        break
    theta+=theta_old #更新梯度参数
    plt.plot([h(xi) for xi in x_test])#将参数运用到测试集里
plt.show()
