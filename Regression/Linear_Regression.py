#!/home/liud/anaconda3/envs/python/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from numpy import linalg
from numpy import corrcoef
from sklearn import linear_model

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
'''
#计算最佳拟合直线,得到模型参数
def standRegress(xList, yList):
	xArr = np.array(xList)
	yArr = np.transpose([yList]) #将yList转化成列向量
	xTx = np.dot(xArr.T, xArr)
	if linalg.det(xTx) == 0: #判断是否为非奇异矩阵
		print "这个矩阵是奇异矩阵，行列式为0"
		return
	ws = np.dot(np.linalg.inv(xTx), np.dot(xArr.T, yArr))
	return ws

#sklearn的写法
def sklearn_standRegress(xList, yList):
	clf = linear_model.LinearRegression(fit_intercept = False) #加载线性回归模型,且让w0 = 0(w0指的是intercept)
	clf.fit(xList, yList) #拟合
	print clf.intercept_
	return np.transpose([clf.coef_]) #返回系数的列向量形式

#展示结果
def show(xList, yList, w):
	import matplotlib.pyplot as plt
	xArr = np.array(xList)
	yArr = np.transpose([yList])
	yPredict = np.dot(xArr, w)
	fig = plt.figure() #创建一幅图
	ax = fig.add_subplot(1, 1, 1)
	print ax
	ax.scatter(xArr[ : , 1].flatten().A[0], yArr[ : , 0].flatten().A[0])
	xCopy = xArr.copy()
	xCopy.sort(0)
	yHat = xCopy * ws
	ax.plot(xCopy[ : , 1], yHat)
	plt.show()

#主函数
def main():
	xList, yList = loadDataSet("/home/liud/PycharmProjects/Machine_Learning/Regression/data/ex0.txt")
	ws = standRegress(xList, yList) #自己按理解实现
	#ws = sklearn_standRegress(xList, yList) #sklearn的实现
	print "最小二乘法得出的回归系数： \n", ws
	show(xList, yList, ws)
	yPredict = np.dot(xList, ws)
	print "相关性： ", corrcoef(yPredict.T.tolist(), yList) #corrcoef中的两个参数尽可能的类型相似,yList是list,因此yPredict是numpy.ndarray且为二维的列向量。

if __name__ == '__main__':
	main()
	print 'Success'