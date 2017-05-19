# -*- coding:utf-8 -*-  # Filename: AdaBoost.py

"""
    AdaBoost�����㷨:(����Ӧboosting)
       �ŵ㣺���������ʵͣ��ױ��룬����Ӧ���ڴ󲿷ַ������ϣ��޲�������
      ȱ�㣺����Ⱥ������


  bagging:�Ծٻ�۷�(bootstrap aggregating)
   ������������س����ķ�������������
      ԭʼ���ݼ�������ѡ��S�εõ�S�������ݼ�����ĥ���㷨�ֱ�������������ݼ�,
     ������ͶƱ��ѡ��ͶƱ���������Ϊ�������

  boosting:������bagging,������������Ͷ�����ͬ��

    boosting�ǹ�ע��Щ���з�������ֵ�����������µķ�������
      bagging���Ǹ�����ѵ���ķ�������������ѵ���ġ�

      boosting������Ȩ�ز���ȣ�Ȩ�ض�Ӧ����һ�ֵ����ɹ���
      bagging������Ȩ�����
"""
from numpy import *


class Adaboosting(object):
    '''�������ݼ�'''
    def loadSimpData(self):
        datMat = matrix(
            [[1., 2.1],
             [2., 1.1],
             [1.3, 1.],
             [1., 1.],
             [2., 1.]])
        classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
        return datMat, classLabels
    '''ͨ����ֵ�ȽϽ��з���
    ����:���ݾ����б꣬��ֵ�����Ⱥţ�lt��gt��
    ���ݷ�Ϊ���ࣨ-1,1������ͬ����ıȽϾ�Ҫ����
    ÿһ��ά�ȵ��������ݸ���ֵ�Ƚϣ����൱���ҵ�һ��ֱ�߻����������ݡ�'''
    def stumpClassify(self, datMat, dimen, threshVal, threshIneq):
        #��������ж���ı�ʶ����
        retArr = ones((shape(datMat)[0], 1))
        if threshIneq == 'lt':
            retArr[datMat[:, dimen] <= threshVal] = -1.0  # С����ֵ���ж�Ϊ-1
        else:
            retArr[datMat[:, dimen] > threshVal] = -1.0  # ������ֵ���ж�Ϊ-1
        return retArr

    '''������������ɺ�����һ��ѭ���ҳ���õķ�������������÷�������Ϣ�����ͷ���Ľ��'''

    def buildStump(self,dataArr,classLables,D):
        dataMatrix = mat(dataArr)
        lableMat = mat(classLables).T
        m, n = shape(dataMatrix)
        numSteps = 10.0  # ������Ӱ����ǵ������������Ʋ���

        bestStump = {}  # �洢����������Ϣ
        bestClassEst = mat(zeros((m, 1)))  # ��õķ�����
        minError = inf  # ����Ѱ����С������

        for i in range(n):
            # ���ÿһ�����ݵ������Сֵ���㲽��
            rangeMin = dataMatrix[:, i].min()
            rangeMax = dataMatrix[:, i].max()
            stepSize = (rangeMax - rangeMin) / numSteps

            for j in range(-1, int(numSteps) + 1):
                threshVal = rangeMin + float(j) * stepSize  # ��ֵ
                for inequal in ['lt', 'gt']:
                    predictedVals = self.stumpClassify(dataMatrix, i, threshVal, inequal)
                    errArr = mat(ones((m, 1)))
                    errArr[predictedVals == lableMat] = 0  # Ϊ1�ı�ʾi�ִ��
                    weightedError = D.T * errArr  # �ִ�ĸ���*Ȩ��(��ʼȨ��=1/M��)���������
                    if weightedError < minError:  # Ѱ����С�ļ�Ȩ������Ȼ�󱣴浱ǰ����Ϣ
                        minError = weightedError
                        bestClassEst = predictedVals.copy()  # ������
                        bestStump['dim'] = i
                        bestStump['thresh'] = threshVal
                        bestStump['ineq'] = inequal
        return bestStump, minError, bestClassEst


    def adaBoostingDs(self, dataArr, classLables, numIt=40):
        '''���ڵ����������AdaBoostingѵ�����̣�'''
        weakClassArr = []  # ��Ѿ���������
        m = shape(dataArr)[0]
        #��ʼ��Ȩ��Ϊ1/m
        D = mat(ones((m, 1)) / m)
        aggClassEst = mat(zeros((m, 1)))
        #����ѵ����ʼ��ѵ��numIt��
        for i in range(numIt):
            bestStump, minError, bestClassEst = self.buildStump(
                dataArr, classLables, D)
            print "bestStump:", bestStump
            print "D:", D.T
            #����÷�������Ȩ��1/2*ln((1-err)/err��
            alpha = float(
                0.5 * log((1.0 - minError) / max(minError, 1e-16)))
            #���÷�������Ϣ�Լ���������Ȩ�ؼ��������������б�
            bestStump['alpha'] = alpha
            weakClassArr.append(bestStump)
            print "alpha:", alpha
            print "classEst:", bestClassEst.T  # ������

            #����ѵ�������ķֲ���Ҳ����Ȩ�صķֲ�
            expon = multiply(-1 * alpha * mat(classLables).T, bestClassEst)
            D = multiply(D, exp(expon))
            D = D / D.sum()

            #���������
            aggClassEst += alpha * bestClassEst
            print "aggClassEst ��", aggClassEst.T
            # �ۼӴ�����
            aggErrors = multiply(sign(aggClassEst) !=
                                 mat(classLables).T, ones((m, 1)))
            # ������ƽ��ֵ
            errorsRate = aggErrors.sum() / m
            print "total error:", errorsRate, "\n"
            if errorsRate == 0.0:
                break
        print "weakClassArr:", weakClassArr
        return weakClassArr


    def adClassify(self, datToClass, classifierArr):
        """
         Ԥ����ࣺ
         datToClass������������
        classifierArr: ѵ���õķ���������
       """
        dataMatrix = mat(datToClass)
        m = shape(dataMatrix)[0]
        aggClassEst = mat(zeros((m, 1)))

        for i in range(len(classifierArr)):  # �ж��ٸ��������������ٴ�
            # ���õ�һ�����������з���
            classEst = self.stumpClassify(dataMatrix, classifierArr[i]['dim'],
                                          classifierArr[i]['thresh'],
                                          classifierArr[i]['ineq']
                                          )
            # alpha ��ʾÿ����������Ȩ�أ�
            print classEst
            aggClassEst += classifierArr[i]['alpha'] * classEst
            print aggClassEst
        return sign(aggClassEst)

if __name__ == "__main__":
    adaboosting = Adaboosting()
    D = mat(ones((5, 1)) / 5)
    dataMat, lableMat = adaboosting.loadSimpData()
    # ѵ��������
    classifierArr = adaboosting.adaBoostingDs(dataMat, lableMat, 40)  # Ԥ������
    result = adaboosting.adClassify([0, 0], classifierArr)
    print result
