import theano
import theano.tensor as T
theano.config.exception_verbosity = 'high'
import numpy as np

#indexing test
def test1():
	a=T.dvector()
	b=a[0:3]
	g=T.sum(b)
	func=theano.function([a], g)
	print func([1,2,3,5,6,7])

test1()