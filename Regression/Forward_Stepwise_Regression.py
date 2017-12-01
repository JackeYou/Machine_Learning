#!/home/liud/anaconda3/envs/python/bin/python
# -*- coding: utf-8 -*-

'''
伪代码：
数据标准化
在每次迭代的过程中：
	设置当前最小的误差lowestError为正无穷(inf)
	对每一个特征值：
		增大或者减小：
			改变一个系数得到一个新的w
			计算新的w下的误差error
			如果error小于当前最小误差lowestError：
				设置wbest等于当前的w
		将w设置为新的wbest
'''
import numpy as np

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

#前向逐步回归
def forwardStepwiseRegression(xArr, yArr, eps=0.01, numIt=100):
	m, n = np.shape(xArr)
	w = np.zeros([n, 1])
	wTotal = np.zeros([numIt, n])
	wTest = np.copy(w)
	wMax = np.copy(w)
	RSS = lambda x, y, w: (np.dot((y - np.dot(x, w)).T, (y - np.dot(x, w))))
	for i in xrange(numIt):
		lowestError = np.inf
		for j in xrange(n):
			for sign in [-1, 1]:
				wTest = np.copy(w)
				wTest[j] += eps * sign
				rssError = RSS(xArr, yArr, wTest)
				if rssError < lowestError:
					lowestError = rssError
					wMax = wTest
		w = np.copy(wMax)
		wTotal[i, :] = w.T
		print '误差 = {}'.format((RSS(xArr, yArr, np.transpose([wTotal[i]]))[0][0]))
	return wTotal
#图像展示
'''
def show(ws):
	plt.plot(ws)
	plt.show()
'''
#主函数
def main():
	xList, yList = loadDataSet("/home/liud/PycharmProjects/Machine_Learning/Regression/data/abalone.txt")
	xArr, yArr = regularize(xList, yList)  # 标准化
	yArr = np.transpose([yArr])
	ws = forwardStepwiseRegression(xArr, yArr)
	print ws
	#show(ws)

if __name__ == '__main__':
	main()
	print "Success"