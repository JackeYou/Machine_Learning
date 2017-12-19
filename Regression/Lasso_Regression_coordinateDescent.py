#!/home/liud/anaconda3/envs/python/bin/python
# -*- coding: utf-8 -*-

'''
	2017.11.30 by youjiaqi
	coordinate descent坐标轴下降法：

'''
#加载的包
import numpy as np
from math import exp
from sklearn import linear_model
from sklearn.base import BaseEstimator, RegressorMixin
from copy import deepcopy
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
def corrCoef(xVector, yVector):
	return (np.dot(xVector.T, yVector)[0][0] / np.shape(xVector)[0] - np.mean(xVector) * np.mean(yVector)) \
		   / ((np.var(xVector) * np.var(yVector)) ** 0.5)

#coordinate descent坐标轴下降法 自己源码实现没想好怎么写
class LassoCD(BaseEstimator, RegressorMixin):
	def __init__(self, alpha = 1.0, max_iter = 1000, fit_intercept = True):
		self.alpha = alpha  #正则化系数
		self.max_iter = max_iter
		self.fit_intercept = fit_intercept
		self.coef_ = None
		self.intercept_ = None

	def _soft_thresholding_operator(self, x, lambda_):
		if x > 0 and lambda_ < abs(x):
			return x - lambda_
		elif x < 0 and lambda_ < abs(x):
			return x + lambda_
		else:
			return 0

	def fit(self, X, y):
		if self.fit_intercept:
			X = np.column_stack((np.ones(len(X)), X))
		beta = np.zeros(X.shape[1])
		if self.fit_intercept:
			beta[0] = np.sum(y - np.dot(X[:, 1:], beta[1:])) / (X.shape[0])

		for iteration in xrange(self.max_iter):
			start = 1 if self.fit_intercept else 0
			for j in xrange(start, len(beta)):
				tmp_beta = deepcopy(beta)
				tmp_beta[j] = 0.0
				r_j = y - np.dot(X, tmp_beta)
				#print r_j
				arg1 = np.dot(X[:, j], r_j)
				arg2 = self.alpha * X.shape[0]
				beta[j] = self._soft_thresholding_operator(arg1, arg2) / (X[:, j] ** 2).sum()
				if self.fit_intercept:
					beta[0] = np.sum(y - np.dot(X[:, 1:], beta[1:])) / (X.shape[0])

		if self.fit_intercept:
			self.intercept_ = beta[0]
			self.coef_ = beta[1:]
		else:
			self.coef_ = beta
		return self

	def predict(self, X):
		y = np.dot(X, self.coef_)
		if self.fit_intercept:
			y += self.intercept_ * np.ones(len(y))
		return y

#调skLearn包的coordinate descent坐标轴下降法
def skLearn_coordinateDescent(xList, yList, lm = 0.2):
	reg = linear_model.Lasso(alpha = lm, fit_intercept = False)
	#print yList.tolist()
	reg.fit(xList, yList)
	return reg.coef_
'''
#lasso测试
def lassoByMeText(xArr, yArr, nTest = 30):
	_, n = np.shape(xArr)
	ws = np.zeros((nTest, n))
	for i in xrange(nTest):
		#自己按照公式实现的
		w = lassoCoordinateDescent(xArr, yArr, lm = exp(i - 10))
		ws[i, :] = w.T
		print('lambda = e^({}), w = {}'.format(i - 10, w.T))
	return ws

#调skLearn包进行测试
def skLearnLassoText(xArr, yArr, nTest = 30):
	_, n = np.shape(xArr)
	ws = np.zeros((nTest, n))
	for i in xrange(nTest):
		# 根据skLearn包进行实现的
		w = skLearn_coordinateDescent(xArr, yArr, lm=exp(i - 10))
		ws[i, :] = w
		print('lambda = e^({}), w = {}'.format(i - 10, w))
	return ws
'''
#主函数
def main():
	xList, yList = loadDataSet("/home/liud/PycharmProjects/Machine_Learning/Regression/data/abalone.txt")
	xArr, yArr = regularize(xList, yList) #标准化
	#yArr = np.transpose([yArr])
	nTest = 30
	_, n = np.shape(xArr)
	ws = np.zeros((nTest, n))
	while (1):
		print '请输入你选择的方式(1.skLearn;2.regression自己实现的岭回归)'
		selectStyle = raw_input()
		if selectStyle == '1':
			for i in xrange(nTest):
				# 根据skLearn包进行实现的
				w = skLearn_coordinateDescent(xArr, yArr, lm=exp(i - 10))
				ws[i, :] = w
				print('lambda = e^({}), w = {}'.format(i - 10, w))
			break
		elif selectStyle == '2':
			for i in xrange(nTest):
				# 自己按照公式实现的
				model = LassoCD(alpha=exp(i - 10), max_iter=1000).fit(xArr, yArr)
				ws[i] = model.coef_
				print('lambda = e^({}), w = {}'.format(i - 10, model.coef_))
			break
		else:
			print '错误输入,请重新输入'
	print ws
	'''
	#对最后结果进行相关系数比较
	yArr_prime = np.dot(xArr, ws)
	corrcoef = corrCoef(yArr, yArr_prime) #可决系数可以作为综合度量回归模型对样本观测值拟合优度的度量指标.
	print'Correlation coefficient:{}'.format(corcoef)
	'''
	#绘制轨迹
	fig = plt.figure()
	ax = fig.add_subplot(111)
	lam = [i - 10 for i in xrange(nTest)]
	ax.plot(lam, ws)
	plt.xlabel('lambda')
	plt.ylabel('ws')
	plt.show()

if __name__ == '__main__':
	main()
	print "Success"