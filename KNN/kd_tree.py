#!/home/liud/anaconda3/envs/python/bin/python
# -*- coding: utf-8 -*-
'''
	kd树：
	即k维树，是一种不断利用数据某个维度划分空间的数据结构；

'''
#加载的包
import numpy as np
import pandas as pd

#利用pandas加载数据
def loadDataSet(filename):
	file = pd.read_table(filename, sep=',',header=None)
	dataArr = np.array(file)
	#进行数据标准化
	#dataArr = regularize(dataArr)
	return dataArr

#标准化数据
def regularize(xArr):
	#数据标准化
	xMean = np.mean(xArr[:, :-1], 0)
	xVar = np.var(xArr[:, :-1], 0)
	xArr[:, :-1] = (xArr[:, :-1] - xMean) / xVar
	return xArr

#kd树类
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

#KDTree的遍历
def preOrder(root):
	print root.dom_elt, root.label
	if root.left:
		preOrder(root.left)
	if root.right:
		preOrder(root.right)

#主函数
def main():
	dataArr = loadDataSet("/home/liud/PycharmProjects/Machine_Learning/KNN/dataTest.txt")
	#传入的数据带有标签
	kd = K_dimensional_Tree(dataArr)
	preOrder(kd.root)

if __name__ == '__main__':
	main()
	print 'Success'