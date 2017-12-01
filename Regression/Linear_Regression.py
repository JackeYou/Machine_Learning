#!/home/liud/anaconda3/envs/python/bin/python
# -*- coding: utf-8 -*-

__import__('os').system('dir')
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from sklearn import linear_model


def loadDataSet(filename):
	numFeat = len(open(filename).readline().split("\t")) - 1
	dataMat = []
	labelMat= []
	fr = open(filename)
	for line in fr.readlines():
		lineArr = []
		curLine = line.strip().split("\t")
		for i in range(numFeat):
			lineArr.append(float(curLine[i]))
		dataMat.append(lineArr)
		labelMat.append(float(curLine[-1]))
	return dataMat, labelMat

#计算最佳拟合直线
def standRegress(xArr, yArr):
	xMat = np.mat(xArr)
	yMat = np.mat(yArr).T
	xTx = xMat.T * xMat
	if linalg.det(xTx) == 0:
		print "这个矩阵是奇异矩阵，行列式为0"
		return
	ws = xTx.I *(xMat.T * yMat)
	return ws

def show():
	xMat = np.mat(xArr)
	yMat = np.mat(yArr)
	yHat = xMat * ws
	fig = plt.figure() #创建一幅图
	ax = fig.add_subplot(1, 1, 1)
	print ax
	ax.scatter(xMat[ : , 1].flatten().A[0], yMat.T[ : , 0].flatten().A[0])
	xCopy = xMat.copy()
	xCopy.sort(0)
	yHat = xCopy * ws
	ax.plot(xCopy[ : , 1], yHat)
	plt.show()

if __name__ == '__main__':
	xArr, yArr = loadDataSet("/home/liud/PycharmProjects/Machine_Learning/Regression/data/ex0.txt")
	# print xArr, yArr
	ws = standRegress(xArr, yArr)
	print "最小二乘法得出的回归系数： \n", ws

	show()
	yHat = np.mat(xArr) * ws
	print "相关性： ", corrcoef(yHat.T, np.mat(yArr))

	clf = linear_model.LinearRegression(fit_intercept=False)
	clf.fit(xArr, yArr)
	print "sklearn 里面线性回归训练得到的回归系数： \n", clf.coef_