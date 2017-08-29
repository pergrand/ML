# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 10:21:05 2016

@author: Administrator
"""
from math import log
import numpy as np

import operator


# 创造测试数据集dataset与标签label
def createDataset():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfaceing', 'flippers']
    return dataSet, labels

# 计算数据集的信息熵（数据集）
# def calcShannonEnt(dataSet):
#     numEntris = len(dataSet)
#     # labelcounts字典键为类别，值为该类别样本数量
#     labelcounts = {}
#     for featVec in dataSet:
#         # 得到dataset中每个样本的最后一个元素，即类别
#         currentlabel = featVec[-1]
#         if currentlabel not in labelcounts:
#             # 当前样本类别labelcounts中没有，添加
#             labelcounts[currentlabel] = 0;
#         # 有则当前样本所属类别数量加一
#         labelcounts[currentlabel] += 1
#     shannonEnt = 0.0
#     # 计算香农熵
#     for key in labelcounts:
#         prob = (float)(labelcounts[key] / numEntris)
#         shannonEnt -= prob * log(prob, 2)
#     return shannonEnt
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for feaVec in dataSet:
        currentLabel = feaVec[-1]
        if currentLabel not in labelCounts:
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt
# 划分数据集（数据集，划分特征索引，特征值）
def spiltDataSet(dataSet, axis, value):
    # python中函数参数按引用传递，因此函数中构建参数的复制
    # 防止传入的参数原值被修改
    retDataSet = []
    for featVec in dataSet:
        if (featVec[axis] == value):
            # 去掉当前这个特征（因为划分中已用过）
            reducedFeatVec = featVec[:axis]  # 拿出该特征之前的特征值
            reducedFeatVec.extend(featVec[axis + 1:])   # 拿出该特征之后的特征值添加到reducedFeatVec集合中
            retDataSet.append(reducedFeatVec)   # 将所有的reducedFeatVec 添加到新的集合中
    return retDataSet

# 选择最好的划分特征（数据集）
def chooseBestFeatureToSplit(dataset):
    # 特征数量
    numFeatures = len(dataset[0]) - 1
    # 原始数据集信息熵
    bestEntropy = calcShannonEnt(dataset)
    # 最优的信息增益
    bestInfoGain = 0.0
    # 最优的特征索引
    bestFeature = -1
    for i in range(numFeatures):
        # 取第i个特征
        featList = [example[i] for example in dataset]
        # set构建集合，将列表中重复元素合并
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            # 按照所取当前特征的不同值划分数据集
            subDataSet = spiltDataSet(dataset, i, value)
            # 计算当前划分的累计香农熵
            prob = len(subDataSet) / float(len(dataset))
            newEntropy += prob * calcShannonEnt(subDataSet)
        # 得到当前特征划分的信息增益
        infoGain = bestEntropy - newEntropy
        # 选出最大的信息增益特征
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

# 若特征用完后仍存在同一分支下有不同类别的样本
# 则此时采用投票方式决定该分支隶属类别
# 即该分支下哪个类别最多，该分支就属哪个类别
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    # 字典排序（字典的迭代器，按照第1个域排序也就是值而不是键，True是降序）
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    # 返回类别
    return sortedClassCount[0][0]

# 递归构建决策树
def creatertree(dataset, labels):
    # 取类别
    classList = [example[-1] for example in dataset]
    # 如果classList中索引为0的类别数量和classList元素数量相等
    # 即分支下都属同一类，停止递归
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 划分类别的特征已用完，停止递归，返回投票结果
    if (len(dataset[0]) == 1):
        return majorityCnt(classList)
    # 选择最具区分度特征
    bestFeat = chooseBestFeatureToSplit(dataset)
    bestFeatLabel = labels[bestFeat]
    # 树用嵌套的字典表示
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataset]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        # 递归构建决策树
        myTree[bestFeatLabel][value] = creatertree(spiltDataSet(dataset, bestFeat, value), subLabels)
    return myTree

# 分类函数（决策树，标签，待分类样本）
def classify(inputTree, featLabels, testVec):
    firstSides = list(inputTree.keys())
    # 找到输入的第一个元素
    firstStr = firstSides[0]
    ##这里表明了python3和python2版本的差别，上述两行代码在2.7中为：firstStr = inputTree.key()[0]
    secondDict = inputTree[firstStr]
    # 找到在label中firstStr的下标
    featIndex = featLabels.index(firstStr)
    # for i in secondDict.keys():
    # print(i)

    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]) == dict:  ###判断一个变量是否为dict，直接type就好
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    ##比较测试数据中的值和树上的值，最后得到节点
    return classLabel

def retrieveTree(i):
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                   {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                   ]
    return listOfTrees[i]


# myDat, labels = createDataset()
# myTree = retrieveTree(0)
# mylabel = classify(myTree, labels, [1, 1])
# print(mylabel)

def main():
    data, label = createDataset()
    myTree = creatertree(data, label)
    mylabel = classify(myTree, ['no surfaceing', 'flippers'], [1, 1])
    print myTree
    print mylabel
if __name__ == '__main__':
    main()