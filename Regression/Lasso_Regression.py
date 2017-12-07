#!/home/liud/anaconda3/envs/python/bin/python
# -*- coding: utf-8 -*-

'''
	2017.11.30 by youjiaqi
	coordinate descent坐标轴下降法：
	整个公式是RSS加上正则项的式子对wk进行求导,在k个特征下进行循环求解wk向量.
	F = ∑(Yi -∑x * wij)^2 + λ * ∑|wj|
	f = のF/のwk = 2∑(Yi -∑xij * wj) * (-xik) + -λ(wk < 0)|[-λ,λ](wk = 0)|λ(wk > 0)
	f = のF/のwk = 2∑(Yi -∑(j!=k)xij * wj)* (-xik) + 2∑xik^2*wk + -λ(wk < 0)|[-λ,λ](wk = 0)|λ(wk > 0)
	另p_k为∑(Yi -∑(j!=k)xij * wj)* (-xik),而z_k为∑xik^2.
	のF/のwk为2z_k * wk - 2p_k + -λ(wk < 0)|[-λ,λ](wk = 0)|λ(wk > 0)。另式子等于0
	最终得出:		 |(p_k + λ/2) / z_k, p_k < - λ/2
			wk = |      0          , - λ/2 < p_k < λ/2
				 |(p_k - λ/2) / z_k, p_k > λ/2
	Least_Angle_Regression最小角回归法

'''
#加载的包
import itertools
import numpy as np
from sklearn import linear_model
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
			p_k = sum([xArr[i, k] * (yArr[i] - sum([xArr[i, j] * w[j, 0]
													for j in xrange(n) if j != k])) for i in xrange(m)])
			#print p_k
			if p_k < -lm / 2:
				w_k = (p_k + lm / 2) / z_k
			elif p_k > lm / 2:
				w_k = (p_k - lm / 2) / z_k
			else:
				w_k = 0
			w[k] = w_k
		print w
		rss_prime = Rss(xArr, yArr, w)
		delta = abs(rss_prime - rss)[0][0]
		#print delta
		rss = rss_prime
		print 'Iteration: {}, delta = {}'.format(it, delta)
		if delta < threshold: #迭代多次满足RSS变化不超过threshold时,迭代停止
			break
	#print w
	return w

#调sklearn包的coordinate descent坐标轴下降法
def skLearn_coordinateDescent(xList, yList, lm = 0.2, threshold = 0.1):
	reg = linear_model.Lasso(alpha = 0.2, fit_intercept = False, max_iter = 10)
	reg.fit(xList, yList)
	print reg.coef_
	return reg.coef_

#Least_Angle_Regression最小角回归法
def lassoLeastAngleRegression(xArr, yArr):
	m, n = np.shape(xArr)
	w = np.zeros((n, 1)) #初始化系数
	Rss = lambda x, y, w: np.dot((y - np.dot(x, w)).T, (y - np.dot(x, w)))
	rss = Rss(xArr, yArr, w)

#主函数
def main():
	xList, yList = loadDataSet("/home/liud/PycharmProjects/Machine_Learning/Regression/data/abalone.txt")
	xArr, yArr = regularize(xList, yList) #标准化
	yArr = np.transpose([yArr])
	#ws = lassoCoordinateDescent(xArr, yArr, 10) #L1正则
	ws = skLearn_coordinateDescent(xArr, yArr, lm = 10)
	print ws
	yArr_prime = np.dot(xArr, ws)
	corcoef = corCoef(yArr, yArr_prime) #可决系数可以作为综合度量回归模型对样本观测值拟合优度的度量指标.
	print'Correlation coefficient:{}'.format(corcoef)

if __name__ == '__main__':
	main()
	print "Success"