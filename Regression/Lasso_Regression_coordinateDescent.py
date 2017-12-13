#!/home/liud/anaconda3/envs/python/bin/python
# -*- coding: utf-8 -*-

'''
	2017.11.30 by youjiaqi
	coordinate descent坐标轴下降法：
	整个公式是RSS加上正则项的式子对wk进行求导,在k个特征下进行循环求解wk向量.
	F = (1/2*m)∑(Yi -∑x * wij)^2 + λ * ∑|wj|
	f = のF/のwk = 1/m * ∑(Yi -∑xij * wj) * (-xik) + -λ(wk < 0)|[-λ,λ](wk = 0)|λ(wk > 0)
	f = のF/のwk = 1/m * ∑(Yi -∑(j!=k)xij * wj)* (-xik) + 1//m * ∑xik^2*wk + -λ(wk < 0)|[-λ,λ](wk = 0)|λ(wk > 0)
	另p_k为∑(Yi -∑(j!=k)xij * wj)* (-xik),而z_k为∑xik^2.
	のF/のwk为1/m * z_k * wk - 1/m * p_k + -λ(wk < 0)|[-λ,λ](wk = 0)|λ(wk > 0)。另式子等于0
	最终得出:		 |(p_k + λ*m) / z_k, p_k < - λm
			wk = |      0          , - λm < p_k < λm
				 |(p_k - λ*m) / z_k, p_k > λm
'''
#加载的包
import itertools
import numpy as np
from math import exp
from sklearn import linear_model
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
def corCoef(xVector, yVector):
	return (np.dot(xVector.T, yVector)[0][0] / np.shape(xVector)[0] - np.mean(xVector) * np.mean(yVector)) \
		   / ((np.var(xVector) * np.var(yVector)) ** 0.5)

#coordinate descent坐标轴下降法
def lassoCoordinateDescent(xArr, yArr, lm = 0.2, threshold = 0.1):
	m, n = np.shape(xArr)
	#print m, n
	w = np.zeros((n, 1)) #初始化回归系数
	Rss = lambda x, y, w: np.dot((y - np.dot(x, w)).T, (y- np.dot(x, w)))
	rss = Rss(xArr, yArr, w)
	#print rss
	niter = itertools.count(1)
	for it in niter: #迭代次数
		for k in xrange(n): #k：特征个数
			z_k = np.dot(xArr[:, k: k + 1].T, xArr[:, k: k + 1])[0][0]
			p_k = sum([xArr[i, k] * (yArr[i, 0] - sum([xArr[i, j] * w[j, 0] for j in xrange(n) if j != k])) for i in xrange(m)])
			#print p_k
			if p_k < -lm * m:
				w_k = (p_k + lm * m) / z_k
			elif p_k > lm * m:
				w_k = (p_k - lm * m) / z_k
			else:
				w_k = 0.0
			w[k] = w_k
		rss_prime = Rss(xArr, yArr, w)
		delta = abs(rss_prime - rss)[0][0]
		rss = rss_prime
		print 'delta = {}'.format(delta)
		print w
		if delta < threshold: #迭代多次满足RSS变化不超过threshold时,迭代停止
			break
	#print w
	return w

#调sklearn包的coordinate descent坐标轴下降法
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

#调sklearn包进行测试
def sklearnLassoText(xArr, yArr, nTest = 30):
	_, n = np.shape(xArr)
	ws = np.zeros((nTest, n))
	for i in xrange(nTest):
		# 根据sklearn包进行实现的
		w = skLearn_coordinateDescent(xArr, yArr, lm=exp(i - 10))
		ws[i, :] = w
		print('lambda = e^({}), w = {}'.format(i - 10, w))
	return ws
'''
#主函数
def main():
	xList, yList = loadDataSet("/home/liud/PycharmProjects/Machine_Learning/Regression/data/abalone.txt")
	xArr, yArr = regularize(xList, yList) #标准化
	yArr = np.transpose([yArr])
	nTest = 30
	_, n = np.shape(xArr)
	ws = np.zeros((nTest, n))
	while (1):
		print '请输入你选择的方式(1.sklearn;2.regression自己实现的岭回归)'
		selectStyle = raw_input()
		if selectStyle == '1':
			for i in xrange(nTest):
				# 根据sklearn包进行实现的
				w = skLearn_coordinateDescent(xArr, yArr, lm=exp(i - 10))
				ws[i, :] = w
				print('lambda = e^({}), w = {}'.format(i - 10, w))
			break
		elif selectStyle == '2':
			for i in xrange(nTest):
				# 自己按照公式实现的
				w = lassoCoordinateDescent(xArr, yArr, lm=exp(i - 10))
				ws[i, :] = w.T
				print('lambda = e^({}), w = {}'.format(i - 10, w.T))
			break
		else:
			print '错误输入,请重新输入'
	#print ws
	'''
	#对最后结果进行相关系数比较
	yArr_prime = np.dot(xArr, ws)
	corcoef = corCoef(yArr, yArr_prime) #可决系数可以作为综合度量回归模型对样本观测值拟合优度的度量指标.
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