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
from sklearn.base import BaseEstimator, RegressorMixin
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

#Least_Angle_Regression最小角回归法,暂时没想好怎么写
class Lasso_LARS(BaseEstimator, RegressorMixin):
	#初始化
	def __init__(self, alpha = 1.0, max_iter = 1000, fit_intercept = True):
		self.alpha = alpha
		self.max_iter = max_iter
		self.fit_intercept = fit_intercept
		self.coef_ = None
		self.intercept_ = None

	#拟合
	def fit(self, xArr, yArr):
		if self.fit_intercept:
			xArr = np.column_stack((np.ones(len(xArr)), xArr))
		beta = np.zeros(np.shape(xArr)[1])

		if self.fit_intercept:
			self.intercept_ = beta[0]
			self.coef_ = beta[1:]
		else:
			self.coef_ = beta
		return self
	def predict(self, xArr):
		pass

#调skLearn的包进行Least_Angle_Regression最小角回归法
def skLearn_LassoLeastAngleRegression(xList, yList, lm = 0.2, threshold = 0.1):
	reg = linear_model.LassoLars(alpha = lm, fit_intercept = False)
	reg.fit(xList, yList)
	return reg.coef_

#主函数
def main():
	xList, yList = loadDataSet("/home/liud/PycharmProjects/Machine_Learning/Regression/data/abalone.txt")
	xArr, yArr = regularize(xList, yList) #标准化
	yArr = np.transpose([yArr])
	nText = 30
	_, n = np.shape(xArr)
	wArr = np.zeros([nText, n])
	while(1):
		print '请输入你选择的方式(1.skLearn;2.regression自己实现的岭回归)'
		selectStyle = raw_input()
		if selectStyle == '1':
			for i in xrange(nText):
				# 根据skLearn调包实现的岭回归
				ws = skLearn_LassoLeastAngleRegression(xArr, yArr, exp(i - 10))
				wArr[i, :] = ws
			break
		elif selectStyle == '2':
			for i in xrange(nText):
				# 自己按理解实现
				model = Lasso_LARS(alpha = exp(i - 10), max_iter = 1000).fit(xArr, yArr)
				wArr[i, :] = model.coef_
			break
		else:
			print '错误输入,请重新输入'
	print wArr

	#绘制轨迹
	fig = plt.figure()
	ax = fig.add_subplot(111)
	lam = [i - 10 for i in xrange(nText)]
	ax.plot(lam, wArr)
	plt.show()

if __name__ == '__main__':
	main()
	print "Success"