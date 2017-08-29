#!/usr/bin/python
# -*- coding:utf-8 -*-
import math
import matplotlib.pyplot as plt
from numpy import *
import random

def main():
#  设置参数
    train,test,alpha,maxCycles,type = getparamenter()
# 加载数据
    trainData,trainLabel = loadDataSet(train)
    testData,testLabel = loadDataSet(test)
# 训练数据
    weights = trainLR(trainData,trainLabel,alpha,maxCycles,type)

    threat = gradAscent(trainData,trainLabel)
    plotBestFit(threat,train)

# 测试数据
    testLR(weights,testData,testLabel)


def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)  # convert to NumPy matrix
    labelMat = mat(classLabels).transpose()  # convert to NumPy matrix

    m, n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n, 1))

    for k in range(maxCycles):  # heavy on matrix operations
        h = sigmoid2(dataMatrix * weights)  # matrix mult
        error = (labelMat - h)  # vector subtraction
        weights = weights + alpha * dataMatrix.transpose() * error  # matrix mult
    return weights

def trainLR(trainData,trainLabel,alpha,maxCycles,type):
    weights = []
    for i in range(0, len(trainData[0]), 1):
        weights.append(1.0)


    for i in range(0, maxCycles, 1):
        errors = []
        if type == "gradDescent":
            for k in range(0, len(trainData), 1):
                result = getMatResult(trainData[k], weights)
                error = trainLabel[k] - sigmoid(result)
                errors.append(error)
            for k in range(0, len(weights), 1):
                updata = 0.0
                for idx in range(0, len(errors), 1):
                    updata += errors[idx] * trainData[idx][k]
                weights[k] += alpha * updata


        elif type == "stocGradDescent":
            for k in range(0, len(trainData), 1):
                result = getMatResult(trainData[k], weights)
                error = trainLabel[k] - sigmoid(result)
                for idx in range(0, len(weights), 1):
                    weights[idx] += alpha * error * trainData[k][idx]

        elif type == "smoothStocGradDescent":
            dataIndex = range(len(trainData))
            for k in range(0, len(trainData), 1):
                randIndex = int(random.uniform(0, len(dataIndex)))
                result = getMatResult(trainData[randIndex], weights)
                error = trainLabel[randIndex] - sigmoid(result)
                for idx in range(0, len(weights), 1):
                    weights[idx] += alpha * error * trainData[randIndex][idx]
        else:
            print "Not support optimize method type!"
    return weights

def getMatResult(data, weights):
    result = 0.0
    for i in range(0, len(data), 1):
        result += data[i] * weights[i]
    return result

# def GetResult():
#     dataMat, labelMat = loadDataSet()
#     weights = trainLR(dataMat, labelMat)
#     print weights

def plotBestFit(weights,train):

    dataMat, labelMat = loadDataSet(train)
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = [];
    ycord1 = []
    xcord2 = [];
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1]);
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1]);
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    #    y=(0.48*x+4.12414)/(0.616)
    #     y = (-weights[0]-weights[1]*x)/weights[2]
    y = (-(float)(weights[0][0]) - (float)(weights[1][0]) * x) / (float)(weights[2][0])
    #y = (-(weights[0][0]) - (weights[1][0]) * x) / (weights[2][0])

    ax.plot(x, y)
    plt.xlabel('X1');
    plt.ylabel('X2');
    plt.show()

def sigmoid(inX):
    return 1.0 / (1 + math.exp(-inX))
def sigmoid2(inX):
    return 1.0 / (1 + exp(-inX))
def testLR(weights, data, label):
    testNum = 0
    matchNum = 0
    for i in range(0, len(data), 1):
        result = getMatResult(data[i], weights)
        predict = 0
        if sigmoid(result) > 0.5:
            predict = 1
        testNum += 1
        if predict == int(label[i]):
            matchNum += 1
    print "testNum:%d\tmatchNum:%d\tratio:%f" % (testNum, matchNum, float(matchNum) / testNum)


def loadDataSet(file):
    data = []
    label = []
    fr = open(file, 'r')
    for line in fr.readlines():
        lineArr = line.strip().split()
        data.append([1.0, float(lineArr[0]), float(lineArr[1])])
        label.append(int(lineArr[2]))
    return data,label
def getparamenter():
    train = 'train_lr.txt'
    test = 'test_lr.txt'
    alpha = 0.01
    maxCycles = 500
    type = 'gradDescent'
    return train,test,alpha,maxCycles,type




if __name__ == '__main__':
    main()