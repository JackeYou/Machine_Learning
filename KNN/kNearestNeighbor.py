#!/home/liud/anaconda3/envs/python/bin/python
# -*- coding: utf-8 -*-
'''
	K近邻算法：
	1.给定一个训练数据集,对新的的输入实例,在训练数据集中找到与该实例最邻近的的K个实例,
	2.这K个实例的多数属于某个类,就把该实例分为这个类;
	KNN 是 supervised learning， non parametric（无参数） instance-based（基于实例） learning algorithm.

	K近邻算法的改进：
	1.不同的K值加权；
	2.距离度量标准根据实际问题，使用不同的距离；
	3.特征归一化；
	4.如果维数过大，可以做PCA降维处理；

	K-近邻的缺陷：
	1.k紧邻算法必须保存全部数据集，如果训练数据集很大，必须使用大量的存储空间；
	2. k近邻算法的另外一个缺陷是它无法给出任何数据的基础结构信息；
	3.无法处理样本的不平衡性
'''
# 加载的包
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.neighbors.classification import KNeighborsClassifier
import matplotlib.pyplot as plt

#加载数据
def loadDataSet(filename):
	name = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
	file = pd.read_table(filename, sep=',',header=None,names=name)
	xArr = np.array(file.ix[:, 0:4])
	yArr = np.array(file['class'])
	return xArr, yArr

#进行交叉验证时所用的加载数据方法
def loadDataSetCV(filename):
	name = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
	file = pd.read_table(filename, sep=',',header=None,names=name)
	xArr = np.array(file.ix[:, 0:4])
	yArr = np.array(file['class'])
	#test_size：如果是浮点数，在0-1之间，表示样本占比；如果是整数的话就是样本的数量；
	#random_state：是随机数的种子；
	return train_test_split(xArr, yArr, test_size=0.33, random_state=40)

#标准化数据
def regularize(xArr):
	#数据标准化
	xMean = np.mean(xArr, 0)
	xVar = np.var(xArr, 0)
	xArr = (xArr - xMean) / xVar
	return xArr

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

#交叉验证：通过交叉验证来完成k值的选择
def cross_validation_Test():
	x_train, x_test, y_train, y_test = loadDataSetCV("/home/liud/PycharmProjects/Machine_Learning/KNN/data.txt")
	k_lst = list(range(1,30))
	lst_scores = []
	for k in k_lst:
		knn = KNeighborsClassifier(n_neighbors=k)
		#cv参数用来规定原始数据分多少份
		#scoring参数用来给模型打分的方式
		scores = cross_val_score(knn, x_train, y_train, cv=10, scoring='accuracy')
		lst_scores.append(np.mean(scores))
	MSE = [1 - x for x in lst_scores]
	optimal_k = k_lst[MSE.index(min(MSE))]
	print "The optimal number of neighbors is %d" % optimal_k
	plt.plot(k_lst, lst_scores)
	plt.xlabel('Number of Neighbors K')
	plt.ylabel('correct classification rate')
	plt.show()

#主函数
def main():
	xArr, yArr = loadDataSet("/home/liud/PycharmProjects/Machine_Learning/KNN/data.txt")
	xArr = regularize(xArr)
	print "请输入test测试样例和k值"
	test = input()
	k = input()
	knn = KnnScratch()
	knn.fit(xArr, yArr)
	result = knn.predict(np.array(test), int(k))
	print result

if __name__ == '__main__':
	main()
	# 交叉验证进行k值选择
	cross_validation_Test()
	print "Success"