#!/home/liud/anaconda3/envs/python/bin/python
# -*- coding: utf-8 -*-
import numpy as np
'''
print "haha"
try:
	a = 10.0
	b = 90.0
	c = float('%.1f' % (a / b * 100))
	print c
except ZeroDivisionError, e:
	raise e
'''
'''
class myException(Exception):
	def __init__(self, message):
		Exception.__init__(self)
		self.message = message

while(1):
	try:
		resultCode = input("shuru:")
		if resultCode < 10:
			print "success"
		else:
			#raise myException('error1')
			raise ZeroDivisionError
	except ZeroDivisionError, e:
		raise e
'''
'''
class Cat(object):
	lia = 10
	def __init__(self, yanse, changdu):
		self.yanse = yanse
		self.changdu = 11
	def getColor(self):
		return self


class myCat(Cat):
	lia = 20
	def __init__(self):
		Cat.__init__(self, "huangse","18")
		self.pifu = "laji"
		#self.yanse = "heise"
		#self.changdu = "20"

a = myCat()
c = a.getColor()
print c.changdu
print a.changdu

'''
#cost = sum(i ** 2 for i in range(3))
#print cost
'''
a = [[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]]
f1 = 1
s1 = 1
print a[f1][(s1 + 2 if s1 + 2 <= 2 else s1 - 1)]
'''