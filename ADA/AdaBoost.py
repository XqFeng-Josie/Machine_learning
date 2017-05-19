# -*- coding:utf-8 -*-  # Filename: AdaBoost.py

"""
    AdaBoost提升算法:(自适应boosting)
       优点：泛化错误率低，易编码，可以应用在大部分分类器上，无参数调整
      缺点：对离群点敏感


  bagging:自举汇聚法(bootstrap aggregating)
   基于数据随机重抽样的分类器构建方法
      原始数据集中重新选择S次得到S个新数据集，将磨沟算法分别作用于这个数据集,
     最后进行投票，选择投票最多的类别作为分类类别

  boosting:类似于bagging,多个分类器类型都是相同的

    boosting是关注那些已有分类器错分的数据来获得新的分类器，
      bagging则是根据已训练的分类器的性能来训练的。

      boosting分类器权重不相等，权重对应与上一轮迭代成功度
      bagging分类器权重相等
"""
from numpy import *


class Adaboosting(object):
    '''加载数据集'''
    def loadSimpData(self):
        datMat = matrix(
            [[1., 2.1],
             [2., 1.1],
             [1.3, 1.],
             [1., 1.],
             [2., 1.]])
        classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
        return datMat, classLabels
    '''通过阈值比较进行分类
    参数:数据矩阵，列标，阈值，不等号（lt，gt）
    数据分为两类（-1,1），不同方向的比较均要进行
    每一个维度的所有数据跟阈值比较，就相当于找到一条直线划分所有数据。'''
    def stumpClassify(self, datMat, dimen, threshVal, threshIneq):
        #声明存放判断完的标识数组
        retArr = ones((shape(datMat)[0], 1))
        if threshIneq == 'lt':
            retArr[datMat[:, dimen] <= threshVal] = -1.0  # 小于阈值的列都为-1
        else:
            retArr[datMat[:, dimen] > threshVal] = -1.0  # 大于阈值的列都为-1
        return retArr

    '''单层决策树生成函数，一次循环找出最好的分类器，返回最好分类器信息，误差，和分类的结果'''

    def buildStump(self,dataArr,classLables,D):
        dataMatrix = mat(dataArr)
        lableMat = mat(classLables).T
        m, n = shape(dataMatrix)
        numSteps = 10.0  # 步数，影响的是迭代次数，控制步长

        bestStump = {}  # 存储分类器的信息
        bestClassEst = mat(zeros((m, 1)))  # 最好的分类器
        minError = inf  # 迭代寻找最小错误率

        for i in range(n):
            # 求出每一列数据的最大最小值计算步长
            rangeMin = dataMatrix[:, i].min()
            rangeMax = dataMatrix[:, i].max()
            stepSize = (rangeMax - rangeMin) / numSteps

            for j in range(-1, int(numSteps) + 1):
                threshVal = rangeMin + float(j) * stepSize  # 阈值
                for inequal in ['lt', 'gt']:
                    predictedVals = self.stumpClassify(dataMatrix, i, threshVal, inequal)
                    errArr = mat(ones((m, 1)))
                    errArr[predictedVals == lableMat] = 0  # 为1的表示i分错的
                    weightedError = D.T * errArr  # 分错的个数*权重(开始权重=1/M行)，计算误差
                    if weightedError < minError:  # 寻找最小的加权错误率然后保存当前的信息
                        minError = weightedError
                        bestClassEst = predictedVals.copy()  # 分类结果
                        bestStump['dim'] = i
                        bestStump['thresh'] = threshVal
                        bestStump['ineq'] = inequal
        return bestStump, minError, bestClassEst


    def adaBoostingDs(self, dataArr, classLables, numIt=40):
        '''基于单层决策树的AdaBoosting训练过程：'''
        weakClassArr = []  # 最佳决策树数组
        m = shape(dataArr)[0]
        #初始化权重为1/m
        D = mat(ones((m, 1)) / m)
        aggClassEst = mat(zeros((m, 1)))
        #迭代训练开始，训练numIt轮
        for i in range(numIt):
            bestStump, minError, bestClassEst = self.buildStump(
                dataArr, classLables, D)
            print "bestStump:", bestStump
            print "D:", D.T
            #计算该分类器的权重1/2*ln((1-err)/err）
            alpha = float(
                0.5 * log((1.0 - minError) / max(minError, 1e-16)))
            #将该分类器信息以及分类器的权重加入弱分类器的列表
            bestStump['alpha'] = alpha
            weakClassArr.append(bestStump)
            print "alpha:", alpha
            print "classEst:", bestClassEst.T  # 类别估计

            #更新训练样本的分布，也就是权重的分布
            expon = multiply(-1 * alpha * mat(classLables).T, bestClassEst)
            D = multiply(D, exp(expon))
            D = D / D.sum()

            #计算误差率
            aggClassEst += alpha * bestClassEst
            print "aggClassEst ；", aggClassEst.T
            # 累加错误率
            aggErrors = multiply(sign(aggClassEst) !=
                                 mat(classLables).T, ones((m, 1)))
            # 错误率平均值
            errorsRate = aggErrors.sum() / m
            print "total error:", errorsRate, "\n"
            if errorsRate == 0.0:
                break
        print "weakClassArr:", weakClassArr
        return weakClassArr


    def adClassify(self, datToClass, classifierArr):
        """
         预测分类：
         datToClass：待分类数据
        classifierArr: 训练好的分类器数组
       """
        dataMatrix = mat(datToClass)
        m = shape(dataMatrix)[0]
        aggClassEst = mat(zeros((m, 1)))

        for i in range(len(classifierArr)):  # 有多少个分类器迭代多少次
            # 调用第一个分类器进行分类
            classEst = self.stumpClassify(dataMatrix, classifierArr[i]['dim'],
                                          classifierArr[i]['thresh'],
                                          classifierArr[i]['ineq']
                                          )
            # alpha 表示每个分类器的权重，
            print classEst
            aggClassEst += classifierArr[i]['alpha'] * classEst
            print aggClassEst
        return sign(aggClassEst)

if __name__ == "__main__":
    adaboosting = Adaboosting()
    D = mat(ones((5, 1)) / 5)
    dataMat, lableMat = adaboosting.loadSimpData()
    # 训练分类器
    classifierArr = adaboosting.adaBoostingDs(dataMat, lableMat, 40)  # 预测数据
    result = adaboosting.adClassify([0, 0], classifierArr)
    print result
