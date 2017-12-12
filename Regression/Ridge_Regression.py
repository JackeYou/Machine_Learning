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
	公式：w = 1/(xTx + λI) * xTy
'''
#加载的包
import matplotlib.pyplot as plt
import numpy as np
from math import exp
from numpy import linalg
from sklearn import linear_model

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
def ridgeRegression(xArr, yArr, ld = 0.2):
	xTx = np.dot(xArr.T, xArr)
	penalty_term = xTx + np.eye(np.shape(xTx)[1]) * ld # 这里面的*运算符对应的是np.multiply()方法，是实现对应元素相乘.
	#print penalty_term
	if np.linalg.det(penalty_term) == 0.0:
		print "This matrix is singular, cannot do inverse"
		return
	w = np.dot(np.linalg.inv(penalty_term), np.dot(xArr.T, yArr))
	return w.T #为了方便加到迭代数组里面,才取的转置.

#调sklearn的岭回归包
def skLearn_ridgeRegression(xList, yList, ld = 0.2):
	reg = linear_model.Ridge(alpha = ld, fit_intercept = False) # intercept要取0,因此数据中没有给出w0所对应的x0，x0 = 1
	reg.fit(xList, yList)
	return reg.coef_

#展示结果
def showRidge(ridgeWeights):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(ridgeWeights)
	plt.show()

def main():
	xList, yList = loadDataSet("/home/liud/PycharmProjects/Machine_Learning/Regression/data/abalone.txt")
	xArr = np.array(xList)
	yArr = np.transpose([yList])  # 加个[]是为了转换成二维
	numTestPts = 30
	# 数据标准化与中心化
	xArr, yArr = regularize(xArr, yArr)
	wArr = np.zeros((numTestPts, np.shape(xArr)[1]))
	while (1):
		print '请输入你选择的方式(1.sklearn;2.regression自己实现的岭回归)'
		selectStyle = raw_input()
		if selectStyle == '1':
			for i in xrange(numTestPts):
				# 根据sklearn调包实现的岭回归
				ws = skLearn_ridgeRegression(xArr, yArr, exp(i - 10))
				wArr[i, :] = ws
			break
		elif selectStyle == '2':
			for i in xrange(numTestPts):
				# 自己按理解实现
				ws = ridgeRegression(xArr, yArr, exp(i - 10))
				wArr[i, :] = ws
			break
		else:
			print '错误输入,请重新输入'
	print wArr
	showRidge(wArr)

if __name__ == '__main__':
	main()
	print 'Success'