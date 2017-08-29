#!/usr/bin/python
import sys

import math

import random
import getopt


def usage():
    print '''Help Information:
    -h, --help: show help information;
    -r, --train: train file;
    -t, --test: test file;
    -k, --ratio: study ratio;
    -i, --iter: iter num;
    -p, --type: optimize type:"gradDescent","stocGradDescent","smoothStocGradDescent";
    '''


def getparamenter():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hr:t:k:i:p:", ["help", "train=", "test=", "kst=", "iter=", "type="])
    except getopt.GetoptError, err:
        print str(err)
        usage()
        sys.exit(1)

    # sys.stderr.write("\ntrain.py : a python script for perception training.\n")
    # sys.stderr.write("Copyright 2016 sxron, search, Sogou. \n")
    # sys.stderr.write("Email: shixiang08abc@gmail.com \n\n")

    train = 'train_lr.txt'
    test = 'test_lr.txt'
    kst = 0.01
    iter = 100
    type = 'gradDescent'
    for i, f in opts:
        if i in ("-h", "--help"):
            usage()
            sys.exit(1)
        elif i in ("-r", "--train"):
            train = f
        elif i in ("-t", "--test"):
            test = f
        elif i in ("-k", "--ratio"):
            kst = float(f)
        elif i in ("-i", "--iter"):
            iter = int(f)
        elif i in ("-p", "--type"):
            type = f
        else:
            assert False, "unknown option"

    print "start trian parameter\ttrain:%s\ttest:%s\tkst:%f\titer:%d\ttype:%s" % (train, test, kst, iter, type)

    return train, test, kst, iter, type


def loadData(file):
    data = []
    label = []
    fin = open(file, 'r')
    while 1:
        line = fin.readline()
        if not line:
            break
        tokens = line.strip().split('\t')
        fea = []
        try:
            lab = float(tokens[-1])
            fea.append(1.0)
            for i in range(0, len(tokens) - 1, 1):
                value = float(tokens[i])
                fea.append(value)
        except:
            continue
        label.append(lab)
        data.append(fea)
    return data, label


def sigmoid(inX):
    return 1.0 / (1 + math.exp(-inX))


def getMatResult(data, weights):
    result = 0.0
    for i in range(0, len(data), 1):
        result += data[i] * weights[i]
    return result


def trainLogRegress(data, label, iter, kst, type):
    weights = []
    for i in range(0, len(data[0]), 1):
        weights.append(1.0)

    for i in range(0, iter, 1):
        errors = []
        if type == "gradDescent":
            for k in range(0, len(data), 1):
                result = getMatResult(data[k], weights)
                error = label[k] - sigmoid(result)
                errors.append(error)
            for k in range(0, len(weights), 1):
                updata = 0.0
                for idx in range(0, len(errors), 1):
                    updata += errors[idx] * data[idx][k]
                weights[k] += kst * updata

        elif type == "stocGradDescent":
            for k in range(0, len(data), 1):
                result = getMatResult(data[k], weights)
                error = label[k] - sigmoid(result)
                for idx in range(0, len(weights), 1):
                    weights[idx] += kst * error * data[k][idx]

        elif type == "smoothStocGradDescent":
            dataIndex = range(len(data))
            for k in range(0, len(data), 1):
                randIndex = int(random.uniform(0, len(dataIndex)))
                result = getMatResult(data[randIndex], weights)
                error = label[randIndex] - sigmoid(result)
                for idx in range(0, len(weights), 1):
                    weights[idx] += kst * error * data[randIndex][idx]
        else:
            print "Not support optimize method type!"
    return weights


def testLogRegress(weights, data, label):
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


def main():
    # set parameter
    train, test, kst, iter, type = getparamenter()

    # load train data
    trnData, trnLabel = loadData(train)
    testData, testLabel = loadData(test)

    # train logregress
    weights = trainLogRegress(trnData, trnLabel, iter, kst, type)

    # test logregress
    testLogRegress(weights, testData, testLabel)


if __name__ == "__main__":
    main()