#!/home/liud/anaconda3/envs/python/bin/python
# -*- coding: utf-8 -*-

#加载的包
import numpy as np
import pandas as pd
from operator import itemgetter

#利用pandas加载数据
def loadDataSet(filename):
	file = pd.read_table(filename, sep=',',header=None)
	dataArr = np.array(file)
	#进行数据标准化
	dataArr = regularize(dataArr)
	return dataArr

#标准化数据
def regularize(xArr):
	#数据标准化
	xMean = np.mean(xArr[:, :-1], 0)
	xVar = np.var(xArr[:, :-1], 0)
	xArr[:, :-1] = (xArr[:, :-1] - xMean) / xVar
	return xArr

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
	k = input("请输入k值:")
	test = input("请输入测试样例：需输入二维数组")
	for i in xrange(np.shape(test)[0]):
		x = test[i]
		queue = BPQ(k)
		ks = knn_search(dataArr)
		Queue = ks.search(kdTree, x, queue)
		print Queue.get_label()

if __name__ == '__main__':
	main()
	print 'Success'