#!/home/liud/anaconda3/envs/python/bin/python
# -*- coding: utf-8 -*-
from math import exp, sqrt

'''
print "haha"
try:
	a = 10.0
	b = 90.0
	c = float('%.1f' % (a / b * 100))
	print c
except ZeroDivisionError, e:
	raise e
'''
'''
class myException(Exception):
	def __init__(self, message):
		Exception.__init__(self)
		self.message = message

while(1):
	try:
		resultCode = input("shuru:")
		if resultCode < 10:
			print "success"
		else:
			#raise myException('error1')
			raise ZeroDivisionError
	except ZeroDivisionError, e:
		raise e
'''
'''
class Cat(object):
	lia = 10
	def __init__(self, yanse, changdu):
		self.yanse = yanse
		self.changdu = 11
	def getColor(self):
		return self.yanse


class myCat(Cat):
	lia = 20
	def __init__(self):
		Cat.__init__(self, "huangse",18)
		self.pifu = "laji"
		#self.yanse = "heise"
		self.changdu = 20

a = myCat()
c = a.getColor()
print c
print a.changdu
'''

#cost = sum(i ** 2 for i in range(3))
#print cost
'''
a = [[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]]
f1 = 1
s1 = 1
print a[f1][(s1 + 2 if s1 + 2 <= 2 else s1 - 1)]
'''

'''
	Dimitri Ambrazis
	Jones Devlin
	Final Project August 2016
	The Coordinate Descent class takes in an excel file, parses it, and performs coordinate descent on the data
	per prior-determined specifications.  The goal is to see what betas are produced.
	Data Key:
		 observations[0] = lcavol
		 observations[1] = lweight
		 observations[2] = age
		 observations[3] = lbph
		 observations[4] = svi
		 observations[5] = lcp
		 observations[6] = gleason
		 observations[7] = pgg45
		 observations[8] = lpsa
'''
'''
import pandas as pd
import numpy as np
from sklearn import linear_model


class CoordinateDescent(object):


	def coordDescent(self, xmatrix, yvalues, lamb, step):
		n = yvalues.size
		p = xmatrix.shape[1]
		betas = np.zeros(shape=(p, 1))
		betasOld = np.zeros(shape=(p, 1))
		maxStep = step

		while maxStep >= step:
			for j in range(p):
				betasOld = betas
				if j == 0:
					betasOld[0] = betas[0]
					betas[0] = self.bnot(xmatrix, yvalues, betas, n, p)
				else:
					betasOld[j] = betas[j]
					betas[j] = self.bother(xmatrix, yvalues, betas, n, p, j, lamb)
			maxStep = self.maxDif(betas, betasOld, p)
		return betas

	#Calculates the beta for j = 0
	def bnot(self, xmatrix, yvalues, betas, n, p):
		outersum = 0
		for i in range(n):
			innersum = 0
			for k in range(p):
				innersum += xmatrix[i, k] * betas[k, 0]
			outersum += yvalues[i] - innersum
		return outersum / n

	#Calculates the beta for j = 1,2,...p
	def bother(self, xmatrix, yvalues, betas, n, p, j, lamb):
		#The next for lines need to be moved out of the loop to improve speed
		denom = 0
		for i in range(n):
			denom += (xmatrix[i, j])**2
		t = lamb / (2 * denom)
		outersum = 0
		for i in range(n):
			innersum = 0
			for k in range(p):
				if k != j:
					innersum += xmatrix[i, k] * betas[k, 0]
			outersum += xmatrix[i, j] * (yvalues[i] - innersum)
		x = outersum / denom
		s = self.shrinkage(x, t)

		return s

	#The shrinkage function
	def shrinkage(self, x, t):
		if x < -t:
			s = x + t
		elif x > t:
			s = x - t
		else:
			s = 0
		return s

	#Finds the maximum absolute difference between two vectors
	def maxDif(self, betas, betasOld, p):
		betadelta = np.zeros(shape=(p, 1))
		for i in range(p):
			betadelta[i] = abs(betas[i] - betasOld[i])
		return np.amax(betadelta)

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

def main(self):
	#prostate = pd.read_table("/home/liud/PycharmProjects/Machine_Learning/Regression/data/abalone.txt")
	#X_df = prostate[["lcavol", "lweight", "age", "lbph", "svi", "lcp", "gleason", "pgg45"]]
	#Y_df = prostate["lpsa"]
	xList, yList = loadDataSet("/home/liud/PycharmProjects/Machine_Learning/Regression/data/abalone.txt")
	x_matrix = np.array(xList)
	y_vector = np.array(yList)

	cd = CoordinateDescent()
	observations = cd.coordDescent(x_matrix, np.transpose([y_vector]), exp(-10), 0.000001)


	print("Lasso coefficients using our implementation")
	print(observations.transpose())
	_lambda = exp(-10)

	clf = linear_model.Lasso(alpha=_lambda, fit_intercept=False, max_iter=50000, tol=0.000001)
	clf.fit(xList, yList)
	print("Lasso coefficients using sklearn.linear_model.Lasso")
	print(np.append(clf.intercept_, clf.coef_))

if __name__ == '__main__':
	main(object)
'''

'''
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from copy import deepcopy
from sklearn import linear_model
import matplotlib.pyplot as plt


class Lasso(BaseEstimator, RegressorMixin):
	def __init__(self, alpha=1.0, max_iter=1000, fit_intercept=True):
		self.alpha = alpha  # 正則化項の係数
		self.max_iter = max_iter  # 繰り返しの回数
		self.fit_intercept = fit_intercept  # 切片(i.e., \beta_0)を用いるか
		self.coef_ = None  # 回帰係数(i.e., \beta)保存用変数
		self.intercept_ = None  # 切片保存用変数

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

		for iteration in range(self.max_iter):
			start = 1 if self.fit_intercept else 0
			for j in range(start, len(beta)):
				tmp_beta = deepcopy(beta)
				tmp_beta[j] = 0.0
				r_j = y - np.dot(X, tmp_beta)
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

#数据标准化和中心化
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

#展示结果
def showLasso(lassoWeights):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(lassoWeights)
	plt.show()

#主函数
def main():
	xList, yList = loadDataSet("/home/liud/PycharmProjects/Machine_Learning/Regression/data/abalone.txt")
	xArr, yArr = regularize(xList, yList)
	nText = 30
	wArr = np.zeros((nText, np.shape(xArr)[1]))
	while(1):
		print '请输入你选择的方式(1.sklearn;2.regression自己实现的岭回归)'
		selectStyle = raw_input()
		if selectStyle == '1':
			for i in xrange(nText):
				model = Lasso(alpha=exp(i - 10), max_iter=1000).fit(xArr, yArr)
				wArr[i] = model.coef_
			break
		elif selectStyle == '2':
			for i in xrange(nText):
				clf = linear_model.Lasso(alpha=0.2, fit_intercept=False, max_iter=1000)
				clf.fit(xArr, yArr)
			break
		else:
			print '错误输入,请重新输入'
	print wArr
	showLasso(wArr)

if __name__ == '__main__':
	main()
	print 'Success'
'''
import numpy as np
import scipy as sc

def cholinsert(R, xArr, X):
	diag_k = np.dot(xArr.T, xArr)
	if np.shape(R) == (0, 0):
		R = np.array([[sqrt(diag_k)]])
	else:
		col_k = np.dot(xArr, X)
		R_k = np.linalg.solve(R, col_k)
		R_kk = sqrt(diag_k - np.dot(R_k.T, R_k))
		R = sc.r_[sc.c_[R,R_k],sc.c_[np.zeros((1,R.shape[0])),R_kk]]
	return R

def lars(X, yArr):
	n, p = np.shape(X)
	ws = np.transpose(np.zeros(n))
	print ws
	act_set = []
	inact_set = range(p)
	beta = np.zeros((p+1, p))
	corr = np.zeros((p+1, p))
	R = np.zeros((0, 0))

	for k in xrange(p):
		print " k = ", k, "active set = ", act_set
		print yArr - ws
		c = np.dot(X.T, yArr - ws)
		print "current correlation = ", c
		corr[k, :] = c

		jMax = inact_set[sc.argmax(np.abs(c[inact_set]))]
		C = c[jMax]
		R = cholinsert(R, X[:, jMax], X[:, act_set])
		act_set.append(jMax)
		inact_set.remove(jMax)

		s = np.sign(c[act_set])
		print s
		s = s.reshape(len(s), 1)
		print "sign = ", s

		GA1 = np.linalg.solve(R, np.linalg.solve(R.T, s))
		AA = 1 / sqrt(sum(GA1 * s))
		w = (AA * GA1)[0][0]

		print "AA =", AA
		print "w = ", w
		u = np.dot(X[:,act_set], w)
		print np.shape(u)

		if k == p:
			print "last variable going all the way to least squares solution"
			gamma = C / AA
		else:
			a = np.dot(X.T, u)
			print np.shape(a)
			print a
			a = a[:, -1]
			print a
			print (C - c[inact_set]) / (AA - a[inact_set])
			print '=================='
			print (C + c[inact_set]) / (AA + a[inact_set])
			tmp = sc.r_[(C - c[inact_set]) / (AA - a[inact_set]),
					 (C + c[inact_set]) / (AA + a[inact_set])]

			gamma = min(sc.r_[tmp[tmp > 0], np.array([C / AA]).reshape(-1)])
		print "ITER k = ", k, ", gamma = ", gamma
		ws = ws + (gamma * u).T[0]
		print ws
		if beta.shape[0] < k:
			beta = sc.c_[beta, np.zeros((beta.shape[0], p))]
		beta[k + 1, act_set] = beta[k, act_set] + gamma * w.T.reshape(-1)

	return beta, corr

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

def main():
	xList, yList = loadDataSet("/home/liud/PycharmProjects/Machine_Learning/Regression/data/abalone.txt")
	xArr, yArr = regularize(xList, yList)  # 标准化
	#yArr = np.transpose([yArr])
	#print yArr
	a, b = lars(xArr, yArr)
	print a,b

if __name__ == '__main__':
	main()
	print '1111'