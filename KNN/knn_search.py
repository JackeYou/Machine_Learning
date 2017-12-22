#!/home/liud/anaconda3/envs/python/bin/python
# -*- coding: utf-8 -*-
'''
	最近邻搜索：
'''
#加载的包
from operator import itemgetter

import numpy as np
import pandas as pd
from kd_tree import K_dimensional_Tree

#利用pandas加载数据
def loadDataSet(filename):
	file = pd.read_table(filename, sep=',',header=None)
	dataArr = np.array(file)
	dataArr = regularize(dataArr)
	return dataArr

#标准化数据
def regularize(xArr):
	#数据标准化
	xMean = np.mean(xArr[:, :-1], 0)
	xVar = np.var(xArr[:, :-1], 0)
	xArr[:, :-1] = (xArr[:, :-1] - xMean) / xVar
	return xArr

#有界优先队列 BPQ
class BPQ(object):
	def __init__(self, k=5, hold_max=False):
		self.data = []
		self.k = k #k近邻的k取值
		self.hold_max = hold_max

	def append(self, point, distance, label):
		self.data.append((point, distance, label))
		self.data.sort(key=itemgetter(1), reverse=self.hold_max)
		self.data = self.data[:self.k]

	#K个最近邻的点
	def get_data(self):
		return [item[0] for item in self.data]

	#统计k个最近邻的点的标签,并返回标签类做多的那个
	def get_label(self):
		labels = [item[2] for item in self.data]
		uniques, counts = np.unique(labels, return_counts=True)
		return uniques[np.argmax(counts)]

	def get_threshold(self):
		return np.inf if len(self.data) == 0 else self.data[-1][1]

	def full(self):
		return len(self.data) >= self.k

#k近邻搜索
class knn_search(object):
	def __init__(self, data):
		self.k = np.shape(data)[1] - 1

	#欧式距离
	def get_distance(self, a, b):
		return np.linalg.norm(np.array(a) - np.array(b))

	#搜索
	def search(self, tree, x, queue):
		if tree != None:
			curDist = self.get_distance(x, tree.dom_elt)
			if curDist < queue.get_threshold():
				queue.append(tree.dom_elt, curDist, tree.label)
			axis = tree.depth % self.k
			search_left = False
			print axis
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
	dataArr = loadDataSet("/home/liud/PycharmProjects/Machine_Learning/KNN/dataTest.txt")
	# 生成kd树
	kd = K_dimensional_Tree(dataArr)
	kdTree = kd.root
	x = input("请输入目标点")
	k = input("请输入k值:")
	queue = BPQ(k)
	ks = knn_search(dataArr)
	Queue = ks.search(kdTree, x, queue)
	print Queue.get_label()

if __name__ == '__main__':
	main()
	print 'Success'