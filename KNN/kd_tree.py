#!/home/liud/anaconda3/envs/python/bin/python
# -*- coding: utf-8 -*-
'''
	kd树：
	即k维树，是一种不断利用数据某个维度划分空间的数据结构；

'''
#加载的包
import numpy as np

#kd树类
class KNode(object):
	def __init__(self, dom_elt, depth, left, right):
		self.dom_elt = dom_elt #样本点
		self.depth = depth #维度
		self.left = left #左子树
		self.right = right #右子树

class K_dimensional_Tree():
	#初始化
	def __init__(self, data):
		self.k = np.shape(data)[1]
		self.root = self.createTree(0, data)
	#构建kd树
	def createTree(self, depth, data_set):
		if not data_set:
			return None
		data_set.sort(key=lambda x: x[depth])
		depth_pos = len(data_set) // 2 #找到中位数的位置
		median = data_set[depth_pos] #中位数分割点
		# 不像公式i = j(mod k) + 1 是因为,x向量是从0开始的,而书上的是从1开始的；
		depth_next = depth % self.k
		return KNode(median, depth,
					 self.createTree(depth_next, data_set[: depth_pos]),
					 self.createTree(depth_next, data_set[depth_pos + 1 :]))

#KDTree的遍历
def preOrder(root):
	print root.dom_elt
	if root.left:
		preOrder(root.left)
	if root.right:
		preOrder(root.right)

#主函数
def main():
	data = [[2,3], [5,4], [9,6], [4,7], [8,1], [7,2]]
	kd = K_dimensional_Tree(data)
	preOrder(kd.root)

if __name__ == '__main__':
	main()
	print 'Success'