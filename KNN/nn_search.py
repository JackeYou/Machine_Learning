#!/home/liud/anaconda3/envs/python/bin/python
# -*- coding: utf-8 -*-

#!/home/liud/anaconda3/envs/python/bin/python
# -*- coding: utf-8 -*-
'''
	最近邻搜索：
'''
#加载的包
import numpy as np

from kd_tree import K_dimensional_Tree

#最近邻搜索
class nn_search(object):
	def __init__(self, tree, x):
		self.tree = tree #生成的kd树
		self.x = x #目标点
		self.k = np.shape([x])[1]
		self.nearestPoint = None #保存最近的点
		self.nearestValue = 0 #保存最近点的值
		self.result = self.travel(tree)

	# 欧氏距离
	def get_distance(self, a, b):
		return np.linalg.norm(np.array(a) - np.array(b))

	#递归搜索
	def travel(self, tree, depth=0):
		if tree != None:
			#递归
			axis = depth % self.k
			if self.x[axis] < tree.dom_elt[axis]:
				self.travel(tree.left, depth + 1)
			else:
				self.travel(tree.right, depth + 1)
			#回溯
			distance = self.get_distance(self.x, tree.dom_elt) #目标点和节点的距离判断
			if self.nearestPoint == None:
				self.nearestPoint = tree.dom_elt
				self.nearestValue = distance
			elif self.nearestValue > distance:
				self.nearestPoint = tree.dom_elt
				self.nearestValue = distance

			print tree.dom_elt, depth, self.nearestValue, tree.dom_elt[axis], self.x[axis], self.nearestPoint
			if abs(self.x[axis] - tree.dom_elt[axis] <= self.nearestValue):
				if self.x[axis] < tree.dom_elt[axis]:
					self.travel(tree.right, depth + 1)
				else:
					self.travel(tree.left, depth + 1)
		return self.nearestPoint

#主函数
def main():
	data = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]
	x = [5, 3] #目标点
	#生成kd树
	kd = K_dimensional_Tree(data)
	kdTree = kd.root
	#进行kd树搜索
	kSearch = nn_search(kdTree, x)
	print kSearch.result

if __name__ == '__main__':
	main()
	print 'Success'