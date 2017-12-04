#!/home/liud/anaconda3/envs/python/bin/python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from math import exp
from numpy import linalg

#加载数据
def loadDataSet(filename):
	xArr = []
	yArr = []
	with open(filename) as f:
		for i in f:
			x = i.rstrip().split("\t")
			#x = i.split("\t")
			x = map(eval, x)
			xArr.append(x[:-1])
			yArr.append(float(x[-1]))
	return xArr ,yArr

'''
	#第二种loadDataSet方法
	numFeature = len(open(filename).readline().split("\t")) - 1
	fr = open(filename)
	for line in fr.readlines():
		lineArr = []
		curLine = line.strip().split("\t")
		#print type(curLine)
		for i in xrange(numFeature):
			lineArr.append(float(curLine[i]))
		xArr.append(lineArr)
		yArr.append(float(curLine[-1]))
	return xArr, yArr
'''

#岭回归公式：w = (xTx + λI).I * xTy
def ridgeRegression(xMat, yMat, ld = 0.2):
	xTx = xMat.T * xMat
	#print np.shape(xTx)
	#print np.shape(xMat)
	penalty_term = np.mat(np.multiply(np.eye(np.shape(xTx)[1]), ld))
	if linalg.det(penalty_term) == 0.0:
		print "This matrix is singular, cannot do inverse"
		return
	w = (xTx + penalty_term).I *(xMat.T *yMat)
	return w

#数据标准化与中心化
def regularize(xMat, yMat):
	#数据标准化
	xMean = np.mean(xMat, 0)
	xVar = np.var(xMat, 0)
	xMat = (xMat - xMean) / xVar
	#数据中心化
	yMean = np.mean(yMat)
	yMat = yMat - yMean
	return xMat, yMat

#测试不同ld的岭回归
def ridgeTest(xArr, yArr):
	xMat = np.mat(xArr)
	yMat = np.mat(yArr).A
	print xMat
	#数据标准化与中心化
	xMat, yMat = regularize(xMat, yMat)
	#print yMat
	numTestPts = 30
	wMat = np.zeros((numTestPts, np.shape(xMat)[1]))
	#print wMat
	for i in xrange(numTestPts):
		ws = ridgeRegression(xMat, yMat, exp(i - 10))
		wMat[i, :] = ws.T
	return wMat

#展示结果
def showRidge():
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(ridgeWeights)
	plt.show()

def main():
	xArr, yArr = loadDataSet("/home/liud/PycharmProjects/Machine_Learning/Regression/data/abalone.txt")
	ridgeWeights = ridgeTest(xArr, yArr)
	print ridgeWeights
	showRidge()

if __name__ == '__main__':
	main()