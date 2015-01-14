import theano
import theano.tensor as T
import theano.ifelse as ifelse
import numpy as np
import pdb
import math,time

import intm
import ac
import acr

if __name__ == '__main__':
	

	if False:
		template = theano.shared(np.float32(np.array([[0.22, 0.44, 0.22],
																							[0.66, 0.88, 0.66],
																							[0.11, 0.33, 0.11]])), name='template')
		image_size = 10
		# transform = np.float32(np.array([[12,-65,-98,27,0.5,12,math.pi/7], [12,-65,-98,27,0.5,12,math.pi/7]]))
		# pdb.set_trace()

		ack = ac.AC(None, template=template, activation=None)
		acker = acr.ACR(ack, theano.shared(image_size, name='image_size'))
		transform = theano.shared(np.float32(np.random.rand(100, 7)), borrow=True)
		igeoPose, intensity = intm.getINTMMatrix(1, None, transform)
		gg=T.grad(T.sum(igeoPose), transform )

	if False:
		transform = T.matrix('transform') 
		getINTM = T.sum(intm.getINTMMatrix(100, None, transform)[0])
		gg = T.grad(getINTM, transform)

		for ii in range(10):
			t1=time.time()
			# print getINTM
			gg.eval({transform: np.float32(np.random.rand(100, 7))})
			print 'time per iter:', time.time() - t1

	#ACR test
	if True:
		image_size = np.float32(28)
		bsize = 1
		rng = np.random.RandomState(123)
		ack = ac.AC(rng, template=None, activation=None)
		acker = acr.ACR(ack, theano.shared(image_size, name='image_size'))

		transform = T.matrix('transform') 
		getINTM = intm.getINTMMatrix(bsize, None, transform)
		# render = acker.render(getINTM[0][0], getINTM[1][0])
		render = acker.render_minibatch(getINTM)

		# gg = T.grad(T.sum(render), transform)#acker.template)
		gg=T.grad(T.sum(render), acker.template)
		for ii in range(10):
			t1=time.time()
			gg.eval({transform: np.float32(np.random.rand(bsize, 7))})
			print 'time per iter:', time.time() - t1
		pdb.set_trace()


	# iGeoPose = ([0][0],
							# intm.getINTMMatrix(1, None, theano.shared(transform))[1][0])
	# render = acker.render(iGeoPose[0], iGeoPose[1])
	# print render.eval()

	# cost = T.sum(T.pow(T.reshape(render, [1, image_size * image_size])
	#                 - theano.shared(np.random.rand(1, image_size * image_size))
	#             , 2))
	# print T.grad(T.sum(render), acker.template).eval()











