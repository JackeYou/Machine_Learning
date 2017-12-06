#!/home/liud/anaconda3/envs/python/bin/python
# -*- coding: utf-8 -*-
'''
	局部加权回归：
	很明显直线非但不能很好的拟合所有数据点,而且误差非常大,但是一条类似二次函数的曲线却能拟合地很好;
	为了解决非线性模型建立线性模型的问题，我们预测一个点的值时,选择与这个点相近的点而不是所有的点做线性回归;
	基于这个思想,便产生了局部加权线性回归算法;在这个算法中，其他离一个点越近,权重越大,对回归系数的贡献就越多.
	θ = (xT*W*x)^(-1) * xT*W*y
	W(i) = exp(-(x(i) - x)^2/2k^2) 为指数衰减函数,其中k为波长参数,控制了权值随距离下降的速率.
	在使用这个算法训练数据的时候,不仅需要学习线性回归的参数,还需要学习波长参数;
	这个算法的问题在于,对于每一个要预测的点,都要重新依据整个数据集计算一个线性回归模型出来,使得算法代价极高.
'''

import numpy as np
from math import exp
from mpmath.matrices import linalg
from numpy import linalg
from numpy.linalg import linalg

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
#局部加权线性线性回归
def lwLR(testPointList, xList, yList, k = 1.0):
	xArr = np.array(xList)
	yArr = np.transpose([yList])
	testPointArr = np.array(testPointList)
	arrDot = lambda x, y: np.dot(x, y) #数组相乘的公式，为了方便单写出来了
	m = xArr.shape[0]
	weights = np.eye(m) #初始化W
	for i in range(m):
		diffArr = np.array(testPointList) - xArr[i]
		weights[i, i] = exp(np.dot(diffArr, np.transpose([diffArr])) / (-2 * k**2)) #上面的公式实现
	xTx = np.dot(xArr.T, np.dot(weights, xArr))
	if linalg.det(xTx) == 0: #判断奇异性
		print "This Matrix is singular, cannot do inverse"
		return
	theta = arrDot(np.linalg.inv(xTx),
				   arrDot(xArr.T,
						  arrDot(weights, yArr))) #上述公式的实现
	return arrDot(testPointArr, theta)[0]

#测试
def lwLRText(textList, xList, yList, k = 1.0):
	m = np.shape(textList)[0] #取出有多少个样本点
	yPredict = np.zeros(m)
	for i in range(m): # 对每个预测点分别进行lwLR，局部加权回归
		yPredict[i] = lwLR(textList[i], xList, yList, k)
	return yPredict

#图像展示
def showLWLR(xList,yList):
	import matplotlib.pyplot as plt
	yHat = lwLRText(xList, xList, yList, 0.03) #第一个xList因为textPoint是xList的一个预测点，而整个过程需要所有的预测点，因此将xList整个列表传给lwLRText方法
	xArr = np.array(xList)
	yArr = np.array(yList)
	sortIndex = np.argsort(xArr[:, 1], 0) #对数据进行返回数组值从小到大的索引值的操作
	xSort = xArr[sortIndex] #根据数组值排序得到的新数组
	#画图
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(xSort[:, 1], yHat[sortIndex]) #拟合的曲线
	ax.scatter(xArr[:, 1], yArr, s = 2, c = 'red')#plot和scatter的方法里面的两个数组都可以用array的一维数组即可，其他人的代码修饰也是为了能够变成这种形式shape(n,) => array
	plt.show()

#主函数
def main():
	xList, yList = loadDataSet("/home/liud/PycharmProjects/Machine_Learning/Regression/data/ex0.txt")
	#print xList[0]
	print "k = 1.0 : ", lwLR(xList[0], xList, yList, k = 1.0)
	print "k = 0.1 : ", lwLR(xList[0], xList, yList, k = 0.1)
	print "k = 0.01 : ", lwLR(xList[0], xList, yList, k = 0.01)
	showLWLR(xList,yList)

if __name__ == '__main__':
	main()
	print 'Success'