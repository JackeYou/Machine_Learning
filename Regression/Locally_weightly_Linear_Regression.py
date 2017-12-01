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

#import demo


def lwlr(testPoint, xArr, yArr, k = 1.0):
	xMat = np.mat(xArr)
	yMat = np.mat(yArr).T
	#zMat = xMat * yMat

	m = shape(xMat)[0]
	weights = np.mat(eye(m))
	for i in range(m):
		diffMat = np.mat(testPoint) - xMat[i, :]
		print xMat[i, :]
		weights[i, i] = exp(diffMat * diffMat.T / (-2 * k**2))
	xTx = np.mat(xMat.T * (weights * xMat))
	if linalg.det(xTx) == 0:
		print "This Matrix is singular, cannot do inverse"
		return
	theta = xTx.I * (xMat.T * (weights * yMat))
	#print (testPoint * theta)[:,0]
	#print type(testPoint * theta)
	return (testPoint * theta)

def lwlrText(textArr, xArr, yArr, k = 1.0):
	m = shape(textArr)[0]
	yHat = np.zeros(m)
	for i in range(m):
		yHat[i] = lwlr(textArr[i], xArr, yArr, k)
	return yHat

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

xArr, yArr = loadDataSet("/home/liud/PycharmProjects/Machine_Learning/Regression/data/ex0.txt")
#xArr, yArr = demo.loadDataSet("/home/liud/桌面/ex0.txt")
print xArr[0]
print "k = 1.0 : ", lwlr(xArr[0], xArr, yArr, k = 1.0)
print "k = 0.1 : ", lwlr(xArr[0], xArr, yArr, k = 0.1)
print "k = 0.01 : ", lwlr(xArr[0], xArr, yArr, k = 0.01)

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

showlwlr()