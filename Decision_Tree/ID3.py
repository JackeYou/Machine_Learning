#!/home/liud/anaconda3/envs/python/bin/python
# -*- coding: utf-8 -*-
'''
	决策树：
	是一种简单但是广泛使用的分类器,通过训练数据构建决策树，可以高效的对未知的数据进行分类；
	优点：计算复杂度不高,输出结果易于理解,对中间值的缺失不敏感,可以处理不相关特征数据;
	缺点：可能会产生过度匹配问题;
'''
#加载的包
import numpy as np
import pandas as pd

from math import log

#利用pandas加载数据
def loadDataSet(filename):
	file = pd.read_table(filename, sep='\t', header=None)
	dataArr = file.values[:,1:]
	return dataArr

#ID3
class ID3(object):
	def __init__(self):
		pass

#计算信息熵
def calShannonEnt(dataSet): #数据集包含类标签列
	numEntries = len(dataSet)
	#创建字典
	labelCount = {}
	for featureVec in dataSet:
		currentLabel = featureVec[-1]
		if currentLabel not in labelCount.keys():
			labelCount[currentLabel] = 0
		labelCount[currentLabel] += 1
	#计算香农熵
	shannonEnt = 0.0
	for key in labelCount:
		prob = float(labelCount[key]) / numEntries #对单个数字取float确保得到小数的后几位
		shannonEnt -= prob * log(prob, 2)
	return shannonEnt

#对离散变量划分数据集,取出该特征取值为value的所有样本
def splitDataSet(dataSet, axis, value): #数据集;特征;分类值
	retDataSet = []
	for featureVec in dataSet:
		if featureVec[axis] == value:
			reduceFeatureVec = featureVec[:axis]
			reduceFeatureVec.extend(featureVec[axis+1:])
			retDataSet.append(reduceFeatureVec)
	return retDataSet #返回不含划分特征的子集

#对连续变量划分数据集，direction规定划分的方向
#决定是划分出小于value的数据样本还是大于value的数据样本集
def splitContinuousDataSet(dataSet, axis, value, direction):
	retDataSet = []
	for featureVec in dataSet:
		if direction == 0:
			if featureVec[axis] > value:
				reduceFeatureVec = featureVec[:axis]
				reduceFeatureVec.extend(featureVec[axis + 1:])
				retDataSet.append(reduceFeatureVec)
		else:
			if featureVec[axis] <= value:
				reduceFeatureVec = featureVec[:axis]
				reduceFeatureVec.extend(featureVec[axis + 1:])
				retDataSet.append(reduceFeatureVec)
	return retDataSet

#选择按照最大信息增益划分数据的函数
def chooseBestFeatureToSplit(dataSet):
	numFeatures = len(dataSet[0]) - 1
	baseEntropy = calShannonEnt(dataSet)
	bestInfoGain = 0.0
	bestFeature = -1
	bestSplitDict = {}
	for i in xrange(numFeatures):
		featureList = [example[i] for example in dataSet]
		#对连续型特征进行处理
		if type(featureList[0]).__name__ == 'float' or type(featureList[0]).__name__ == 'int':
			pass
		#对离散型特征进行处理
		else:
			pass



def main():
	dataArr = loadDataSet("/home/liud/PycharmProjects/Machine_Learning/Decision_Tree/data.txt")

if __name__ == '__main__':
	main()
	print 'Success'