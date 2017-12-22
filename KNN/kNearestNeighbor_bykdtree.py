#!/home/liud/anaconda3/envs/python/bin/python
# -*- coding: utf-8 -*-

#加载的包
import numpy as np
import pandas as pd
from operator import itemgetter
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.neighbors.classification import KNeighborsClassifier
import matplotlib.pyplot as plt

#利用pandas加载数据
def loadDataSet(filename):
	file = pd.read_table(filename, sep=',',header=None)
	dataArr = np.array(file)
	#进行数据标准化
	dataArr = regularize(dataArr)
	return dataArr

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
	xMean = np.mean(xArr[:, :-1], 0)
	xVar = np.var(xArr[:, :-1], 0)
	xArr[:, :-1] = (xArr[:, :-1] - xMean) / xVar
	return xArr

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
	#准确率
	optimal_k = k_lst[lst_scores.index(max(lst_scores))]
	print "The optimal number of neighbors is %d" % optimal_k
	plt.plot(k_lst, lst_scores)
	plt.xlabel('Number of Neighbors K')
	plt.ylabel('correct classification rate')
	plt.show()
	return optimal_k

#kd树 树节点类
class KNode(object):
	def __init__(self, dom_elt, label,depth, left, right):
		self.dom_elt = dom_elt #样本点
		self.label = label #样本点的类别
		self.depth = depth #维度
		self.left = left #左子树
		self.right = right #右子树

#kd树的构建(递归法)
class K_dimensional_Tree():
	#初始化
	def __init__(self, data):
		#因为数据还有标签,所以求数据维度要减去1
		self.k = np.shape(data)[1] - 1
		self.root = self.createTree(0, data)

	#构建kd树
	def createTree(self, depth, data_set):
		if np.shape(data_set)[0] < 1:
			return None
		else:
			# 公式i = j(mod k) + 1；
			# 为了防止数据越界而将此行放入最前面；
			axis = depth % self.k #axis代表x的维度
			sortIndex = data_set[:, axis].argsort() #按照某一axis维度进行排序,而axis的取值取决于上述公式
			sortData = data_set[sortIndex][:, :-1] #对应的x数据进行按axis维度排序
			sortLabel = data_set[sortIndex][:, -1] #相应的标签也跟着变化
			depth_pos = len(data_set) // 2 #找到中位数的位置
			median = sortData[depth_pos] #中位数分割点
			medianLabel = sortLabel[depth_pos] #中位数分割点的标签
			return KNode(median, medianLabel, depth,
						 self.createTree(axis + 1, data_set[sortIndex][: depth_pos]),
						 self.createTree(axis + 1, data_set[sortIndex][depth_pos + 1 :]))

# 有界优先队列 BPQ
class BPQ(object):
	def __init__(self, k=5, hold_max=False):
		self.data = []
		self.k = k  # k近邻的k取值
		self.hold_max = hold_max

	def append(self, point, distance, label):
		self.data.append((point, distance, label))
		self.data.sort(key=itemgetter(1), reverse=self.hold_max)
		self.data = self.data[:self.k]

	# K个最近邻的点
	def get_data(self):
		return [item[0] for item in self.data]

	# 统计k个最近邻的点的标签,并返回标签类做多的那个
	def get_label(self):
		labels = [item[2] for item in self.data]
		uniques, counts = np.unique(labels, return_counts=True)
		return uniques[np.argmax(counts)]

	def get_threshold(self):
		return np.inf if len(self.data) == 0 else self.data[-1][1]

	def full(self):
		return len(self.data) >= self.k

# k近邻搜索
class knn_search(object):
	def __init__(self, data):
		self.k = np.shape(data)[1] - 1

	# 欧式距离
	def get_distance(self, a, b):
		return np.linalg.norm(np.array(a) - np.array(b))

	# 搜索
	def search(self, tree, x, queue):
		if tree != None:
			curDist = self.get_distance(x, tree.dom_elt)
			if curDist < queue.get_threshold():
				queue.append(tree.dom_elt, curDist, tree.label)
			axis = tree.depth % self.k
			search_left = False
			# print axis
			if x[axis] < tree.dom_elt[axis]:
				search_left = True
				queue = self.search(tree.left, x, queue)
			else:
				queue = self.search(tree.right, x, queue)

			if not queue.full() or np.abs(tree.dom_elt[axis] - x[axis]) < queue.get_threshold():
				if search_left:
					queue = self.search(tree.right, x, queue)
				else:
					queue = self.search(tree.left, x, queue)
		return queue

#主函数
def main():
	# 生成的数据带有标签
	dataArr = loadDataSet("/home/liud/PycharmProjects/Machine_Learning/KNN/data.txt")
	# 生成kd树
	kd = K_dimensional_Tree(dataArr)
	kdTree = kd.root
	#对生成的kd树进行相应的搜索
	test = input("请输入测试样例：需输入二维数组")
	k = cross_validation_Test() #通过交叉验证进行k值的选择
	#按照目标样本的个数进行搜索其k个最近的样本点,并输出其目标样本的类别(k个样本类别的最多类别为其真实类别)
	for i in xrange(np.shape(test)[0]):
		x = test[i] #单个目标样本点
		queue = BPQ(k)
		#根据目标点来进行搜索其最近的k个样本点
		ks = knn_search(dataArr)
		Queue = ks.search(kdTree, x, queue)
		print Queue.get_label()

if __name__ == '__main__':
	main()
	print 'Success'