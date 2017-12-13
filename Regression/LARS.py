#!/home/liud/anaconda3/envs/python/bin/python
# -*- coding: utf-8 -*-
'''
	Least_Angle_Regression最小角回归法

'''
#加载的包
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

#Least_Angle_Regression最小角回归法
def LeastAngleRegression(xArr, yArr, lm=0.2, nSteps=350):
	nrows, ncols = np.shape(xArr)
	# initialize a vector of coefficients beta
	beta = [0.0] * ncols
	# initialize matrix of betas at each step
	betaMat = []
	betaMat.append(list(beta))
	# number of steps to take
	nSteps = 350
	stepSize = 0.004
	for i in range(nSteps):
		# calculate residuals
		residuals = [0.0] * nrows
		for j in range(nrows):
			yHat = sum([xArr[j][k] * beta[k] for k in range(ncols)])
			residuals[j] = yArr[j] - yHat
		# calculate correlation between attribute columns from
		# normalized wine and residual
		corr = [0.0] * ncols
		for j in range(ncols):
			corr[j] = sum([xArr[k][j] * residuals[k] for k in range(nrows)]) / nrows
		iStar = 0
		corrStar = corr[0]
		for j in range(1, (ncols)):
			if abs(corrStar) < abs(corr[j]):
				iStar = j; corrStar = corr[j]
		beta[iStar] += stepSize * corrStar / abs(corrStar)
		betaMat.append(list(beta))
		print beta

	for i in range(ncols):
		# plot range of beta values for each attribute
		coefCurve = [betaMat[k][i] for k in range(nSteps)]
		xaxis = range(nSteps)
		plt.plot(xaxis, coefCurve)

	plt.xlabel("Steps Taken")
	plt.ylabel(("Coefficient Values"))
	plt.show()

#主函数
def main():
	xList, yList = loadDataSet("/home/liud/PycharmProjects/Machine_Learning/Regression/data/abalone.txt")
	xArr, yArr = regularize(xList, yList) #标准化
	yArr = np.transpose([yArr])
	LeastAngleRegression(xArr, yArr, 0.004, 350)

if __name__ == '__main__':
	main()
	print "Success"