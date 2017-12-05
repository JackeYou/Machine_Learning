#!/home/liud/anaconda3/envs/python/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from math import exp
from mpmath.matrices import linalg
from numpy import linalg
from numpy.linalg import linalg
from numpy.ma.core import shape
from numpy.matlib import eye
import matplotlib.pyplot as plt

#加载数据
def loadDataSet(filename):
	xList = []
	yList = []
	with open(filename) as fn:
		for i in fn:
			x = i.rstrip().split("\t")
			#x = map(eval, x)  此函数eval容易造成恶意输入
			x = map(eval, x)
			xList.append(x[: -1])
			yList.append(float(x[-1]))
	return xList, yList
'''
def loadDataSet(filename):
	numFeature = len(open(filename).readline().split("\t")) - 1
	dataMat = []
	labelMat = []
	fr = open(filename)
	#print fr.readlines()
	for line in fr.readlines():
		lineArr = []
		print line
		curArr = line.strip().split("\t")
		for i in range(numFeature):
			lineArr.append(float(curArr[i]))
			#print lineArr
		dataMat.append(lineArr)
		labelMat.append(float(curArr[-1]))
		#print labelMat
	return dataMat, labelMat
'''

def lwLR(testPointList, xList, yList, k = 1.0):
	xArr = np.array(xList)
	yArr = np.transpose([yList])
	testPointArr = np.array(testPointList)
	arrDot = lambda x, y: np.dot(x, y)
	m = xArr.shape[0]
	weights = np.eye(m)
	for i in range(m):
		diffArr = np.array(testPointList) - xArr[i]
		weights[i, i] = exp(np.dot(diffArr, np.transpose([diffArr])) / (-2 * k**2))
	xTx = np.dot(xArr.T, np.dot(weights, xArr))
	if linalg.det(xTx) == 0:
		print "This Matrix is singular, cannot do inverse"
		return
	theta = arrDot(np.linalg.inv(xTx),
				   arrDot(xArr.T,
						  arrDot(weights, yArr)))
	return arrDot(testPointArr, theta)[0]

def lwlrText(textArr, xArr, yArr, k = 1.0):
	m = shape(textArr)[0]
	yHat = np.zeros(m)
	for i in range(m):
		yHat[i] = lwLR(textArr[i], xArr, yArr, k)
	return yHat

#图像展示
def showlwlr():
	yHat = lwlrText(xArr, xArr, yArr, 0.03)
	xMat = np.mat(xArr)
	sortIndex = xMat[:, 1].argsort(0)
	xSort = xMat[sortIndex][:, 0, :]
	#print xMat[sortIndex]
	#print xSort
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(xSort[:,1],yHat[sortIndex])
	ax.scatter(xMat[:, 1].flatten().A[0],np.mat(yArr).T[:, 0].flatten().A[0],s = 2,c = 'red')
	plt.show()

#主函数
def main():
	xList, yList = loadDataSet("/home/liud/PycharmProjects/Machine_Learning/Regression/data/ex0.txt")
	# xArr, yArr = demo.loadDataSet("/home/liud/桌面/ex0.txt")
	print xList[0]
	print "k = 1.0 : ", lwLR(xList[0], xList, yList, k = 1.0)
	print "k = 0.1 : ", lwLR(xList[0], xList, yList, k = 0.1)
	print "k = 0.01 : ", lwLR(xList[0], xList, yList, k = 0.01)
	showlwlr()

if __name__ == '__main__':
	main()
	print 'Success'