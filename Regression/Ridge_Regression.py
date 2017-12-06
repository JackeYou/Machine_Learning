#!/home/liud/anaconda3/envs/python/bin/python
# -*- coding: utf-8 -*-

'''
	岭回归：
	有些情况下无法按照上面的典型回归的方法去训练模型.
	比如，训练样本数量少,甚至少于样本维数,这样将导致数据矩阵无法求逆；
	又比如样本特征中存在大量相似的特征,导致很多参数所代表的意义重复.
	总得来说，就是光靠训练样本进行无偏估计是不好用了.
	这个时候，我们就应用结构风险最小化的模型选择策略，在经验风险最小化的基础上加入正则化因子.
	当正则化因子选择为模型参数的二范数的时候,整个回归的方法就叫做岭回归.
	w = 1/(xTx + λI) * xTy
'''
import matplotlib.pyplot as plt
import numpy as np
from math import exp
from numpy import linalg

#加载数据
def loadDataSet(filename):
	xList = []
	yList = []
	with open(filename) as f:
		for i in f:
			x = i.rstrip().split("\t")
			#x = i.split("\t")
			x = map(eval, x)
			xList.append(x[:-1])
			yList.append(float(x[-1]))
	return xList ,yList

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

#数据标准化与中心化
def regularize(xArr, yArr):
	#数据标准化
	xMean = np.mean(xArr, 0)
	xVar = np.var(xArr, 0)
	xArrRL = (xArr - xMean) / xVar
	#数据中心化
	yMean = np.mean(yArr)
	yArrCen = yArr - yMean
	return xArrRL, yArrCen

#岭回归公式：w = (xTx + λI).I * xTy
def ridgeRegression(xArr, yArr, ld=0.2):
	xTx = np.dot(xArr.T, xArr)
	penalty_term = xTx + np.eye(np.shape(xTx)[1]) * ld # 这里面的*运算符对应的是np.multiply()方法，是实现对应元素相乘.
	#print penalty_term
	if np.linalg.det(penalty_term) == 0.0:
		print "This matrix is singular, cannot do inverse"
		return
	w = np.dot(np.linalg.inv(penalty_term), np.dot(xArr.T, yArr))
	return w

#测试不同ld的岭回归
def ridgeTest(xList, yList, numTestPts = 30):
	xArr = np.array(xList)
	yArr = np.transpose([yList]) #加个[]是为了转换成二维
	#数据标准化与中心化
	xArr, yArr = regularize(xArr, yArr)
	wArr = np.zeros((numTestPts, np.shape(xArr)[1]))
	for i in xrange(numTestPts):
		ws = ridgeRegression(xArr, yArr, exp(i - 10))
		wArr[i, :] = ws.T
		#print wArr[i]
	return wArr

#展示结果
def showRidge(ridgeWeights):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(ridgeWeights)
	plt.show()

def main():
	xList, yList = loadDataSet("/home/liud/PycharmProjects/Machine_Learning/Regression/data/abalone.txt")
	ridgeWeights = ridgeTest(xList, yList)
	print ridgeWeights
	showRidge(ridgeWeights)

if __name__ == '__main__':
	main()
	print 'Success'