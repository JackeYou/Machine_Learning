#!/home/liud/anaconda3/envs/python/bin/python
# -*- coding: utf-8 -*-
'''
	Least_Angle_Regression最小角回归法
'''
#加载的包
import itertools
import numpy as np
from math import exp
from sklearn import linear_model
import matplotlib.pyplot as plt
#加载数据
def loadDataSet(filename):
	xList = []
	yList = []
	with open(filename) as fn:
		for i in fn:
			x = i.rstrip().split("\t")
			x = map(eval, x)
			xList.append(x[: -1])
			yList.append(float(x[-1]))
	return xList, yList

#标准化数据
def regularize(xList, yList):
	xArr = np.array(xList)
	yArr = np.array(yList)
	#数据标准化
	xMean = np.mean(xArr, 0)
	xVar = np.var(xArr, 0)
	xArr = (xArr - xMean) / xVar
	#数据中心化,去除截距项的影响
	yMean = np.mean(yArr)
	yArr = yArr - yMean
	return xArr, yArr

#相关系数
def corCoef(xVector, yVector):
	return (np.dot(xVector.T, yVector)[0][0] / np.shape(xVector)[0] - np.mean(xVector) * np.mean(yVector)) \
		   / ((np.var(xVector) * np.var(yVector)) ** 0.5)

#Least_Angle_Regression最小角回归法
def lassoLeastAngleRegression(xArr, yArr, numIt=100):
	m, n = np.shape(xArr)
	w = np.zeros((n, 1)) #初始化系数
	wTotal = np.zeros([numIt, n])
	wMax = np.copy(w)
	Rss = lambda x, y, w: np.dot((y - np.dot(x, w)).T, (y - np.dot(x, w)))
	for i in xrange(numIt):
		lowestError = np.inf
		for j in xrange(n):
			pass

#调sklearn的包进行Least_Angle_Regression最小角回归法
def sklearn_LassoLeastAngleRegression(xList, yList, lm = 0.2, threshold = 0.1):
	reg = linear_model.LassoLars(alpha = .1, fit_intercept = False)
	reg.fit(xList, yList)
	return reg.coef_

#主函数
def main():
	xList, yList = loadDataSet("/home/liud/PycharmProjects/Machine_Learning/Regression/data/abalone.txt")
	xArr, yArr = regularize(xList, yList) #标准化
	yArr = np.transpose([yArr])
	

	#绘制轨迹
	fig = plt.figure()
	ax = fig.add_subplot(111)
	lam = [i - 10 for i in xrange(nTest)]
	ax.plot(lam, ws)
	plt.show()

if __name__ == '__main__':
	main()
	print "Success"