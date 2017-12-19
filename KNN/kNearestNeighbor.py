#!/home/liud/anaconda3/envs/python/bin/python
# -*- coding: utf-8 -*-
'''
	K近邻算法：
	1.给定一个训练数据集,对新的的输入实例,在训练数据集中找到与该实例最邻近的的K个实例,
	2.这K个实例的多数属于某个类,就把该实例分为这个类;
	KNN 是 supervised learning， non parametric（无参数） instance-based（基于实例） learning algorithm.
'''
# 加载的包
from collections import Counter

import numpy as np
import pandas as pd


#加载数据
def loadDataSet(filename):
	name = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
	file = pd.read_table(filename, sep=',',header=None,names=name)
	xArr = np.array(file.ix[:, 0:4])
	yArr = np.array(file['class'])
	return xArr, yArr

class KnnScratch(object):
	#拟合
	def fit(self, x_train, y_train):
		self.x_train = x_train
		self.y_train = y_train
	#样本预测
	def predict_once(self, x_test, k):
		lst_distance = {}
		lst_predict = []
		for i in xrange(len(self.x_train)):
			# euclidean distance欧几里得距离：
			distance = np.linalg.norm(x_test - self.x_train[i, :])
			lst_distance[i] = distance
		# 排序并生成键值对
		lst_distance = sorted(lst_distance.iteritems(), key=lambda a:a[1], reverse=False)
		for i in xrange(k):
			index = lst_distance[i][0]
			lst_predict.append(self.y_train[index])
		#print Counter(lst_predict).most_common(1)
		#collections的counter类是一个跟踪值出现的次数的容器
		return Counter(lst_predict).most_common(1)[0][0]
	#多样本预测
	def predict(self, x_test, k):
		lst_predict = []
		for i in xrange(len(x_test)):
			lst_predict.append(self.predict_once(x_test[i, :], k))
		return lst_predict

#主函数
def main():
	xArr, yArr = loadDataSet("/home/liud/PycharmProjects/Machine_Learning/KNN/data.txt")
	print "请输入test测试样例和k值"
	test = input()
	k = input()
	knn = KnnScratch()
	knn.fit(xArr, yArr)
	result = knn.predict(np.array(test), int(k))
	print result

if __name__ == '__main__':
	main()
	print "Success"