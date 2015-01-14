import theano
import theano.tensor as T
# theano.config.exception_verbosity = 'high'

import gpnn
import intm
import ac
import acr
import om
import numpy as np
import pdb,time,math
from logistic_sgd import LogisticRegression, load_data

global image_size
image_size = 28

#xx={self.x:np.float32(np.random.rand(2,28*28))}

class CapsuleNetwork(object):
	def __init__(self, dataset='mnist.pkl.gz'):
		self.num_acrs = 1#9 #number of ACRs
		self.rng = np.random.RandomState(123)
		# self.theano_rng = RandomStreams(rng.randint(2 ** 30))
		self.batch_size = 20#20
		# if True:

		datasets = load_data(dataset)
		self.train_set_x, self.train_set_y = datasets[0]
		self.valid_set_x, self.valid_set_y = datasets[1]
		self.test_set_x,  self.test_set_y = datasets[2]

		# compute number of minibatches for training, validation and testing
		self.n_train_batches = self.train_set_x.get_value(borrow=True).shape[0] / self.batch_size
		self.n_valid_batches = 10
		# self.n_valid_batches = self.valid_set_x.get_value(borrow=True).shape[0] / self.batch_size
		self.n_test_batches = 100
		# self.n_test_batches = self.test_set_x.get_value(borrow=True).shape[0] / self.batch_size
		self.index = T.lscalar()  # index to a [mini]batch
		self.x = T.matrix('x')

		self.image_size = int(T.shape(self.train_set_x[1]).eval()**(1./2))
		print "image size: ", self.image_size
		# else:
		# 	# allocate symbolic variables for the data
		# 	self.index = T.lscalar()    # index to a [mini]batch
		# 	#self.x = T.matrix('x')  # the data is presented as rasterized images
		# self.x = theano.shared(np.random.rand(self.batch_size,image_size*image_size))


	def create_model(self, L1_reg=0.00, L2_reg=0.0001):
		xx={self.x:np.float32(np.random.rand(self.batch_size,28*28))}

		self.encoder = gpnn.GPNN(rng=self.rng,
														input=self.x,
														n_in=self.image_size * self.image_size,
														n_hidden=20, n_out=self.num_acrs*7)
		# import time
		# for ii in range(100):
		# 	t1=time.time()
		# 	xx={self.x:np.float32(np.random.rand(self.batch_size,28*28))}
		# 	gg=T.grad(T.sum(self.encoder.output), self.encoder.params[0]).eval(xx)
		# 	print 'time per iter:', time.time() - t1
		# pdb.set_trace()


		self.iGeoArray = dict()
		self.ACRArray = []
		self.outputs = []
		for i in range(self.num_acrs):
			igeon_indx = range(i,i+7) #pose + intensity
			# self.iGeoArray[i] = intm.getINTMMatrix(self.x, self.batch_size,self.rng, self.encoder.output[:,igeon_indx])
			self.iGeoArray[i] = intm.getINTMMatrix(self.batch_size,self.rng, self.encoder.output[:,igeon_indx])
			# template = theano.shared(np.array([[0.22, 0.44, 0.22],
		 #                                     [0.66, 0.88, 0.66],
		 #                                     [0.11, 0.33, 0.11]]))

			ack = ac.AC(self.rng, template=None, activation=None)
			self.ACRArray.append(acr.ACR(ack, self.image_size))
			# pdb.set_trace()
			self.outputs.append(self.ACRArray[i].render_minibatch((self.iGeoArray[i][0], self.iGeoArray[i][1])))

		# renderCache = self.outputs[0]
		# for i in range(1,self.num_acrs):
		# 	renderCache = T.stack(renderCache, self.outputs[i])
		renderCache = T.zeros([self.num_acrs, self.batch_size, self.image_size, self.image_size ])
		for i in range(self.num_acrs):
			renderCache = T.set_subtensor(renderCache[i,:,:,:], self.outputs[i])
	
		#combine capsule ACRs
		rendering = om.om(renderCache)

		#define cost function
		self.cost = T.pow(T.reshape(rendering, [self.batch_size, self.image_size * self.image_size]) - self.x, 2)
		self.cost = T.sum(self.cost)
		#adding regularization
		self.cost = self.cost + L1_reg*(self.encoder.L1) + L2_reg*(self.encoder.L2_sqr)

		#aggregate all params
		self.params = self.encoder.params
		for acker in self.ACRArray:
			self.params = self.params + acker.ac.params
		# self.params = [self.params[0]]
		#xx={self.x:np.float32(np.random.rand(self.batch_size,28*28))}


def train_test(learning_rate=0.01, n_epochs=1, dataset='mnist.pkl.gz'):
	######################
	# BUILD ACTUAL MODEL #
	######################
	print '... building the model'
	learning_rate = np.float32(learning_rate)

	cnet = CapsuleNetwork(dataset='mnist.pkl.gz')
	print 'defined CapsuleNetwork'
	cnet.create_model()
	print 'created model'
	# compiling a Theano function that computes the mistakes that are made
	# by the model on a minibatch
	# pdb.set_trace()

	test_model = theano.function(inputs=[cnet.index],
					outputs=cnet.cost,
					givens={
							cnet.x: cnet.test_set_x[cnet.index * cnet.batch_size:(cnet.index + 1) * cnet.batch_size]})

	validate_model = theano.function(inputs=[cnet.index],
					outputs=cnet.cost,
					givens={
							cnet.x: cnet.valid_set_x[cnet.index * cnet.batch_size:(cnet.index + 1) * cnet.batch_size]})

	print 'created test and validate functions'

	# compute the gradient of cost with respect to theta (stored in params)
	# the resulting gradients will be stored in a list gparams
	gparams = []
	for param in cnet.params:
			gparam = T.grad(cnet.cost, param)
			gparams.append(gparam)
	# gparams = T.grad(cnet.cost, cnet.params)

	print 'built gparams'

	# specify how to update the parameters of the model as a list of
	# (variable, update expression) pairs
	updates = []
	for param, gparam in zip(cnet.params, gparams):
			updates.append((param, param - learning_rate * gparam))

	print 'built updates'
	# compiling a Theano function `train_model` that returns the cost, but
	# in the same time updates the parameter of the model based on the rules
	# defined in `updates`
	train_model = theano.function(inputs=[cnet.index], outputs=cnet.cost,
					updates=updates,
					givens={
							cnet.x: cnet.train_set_x[cnet.index * cnet.batch_size:(cnet.index + 1) * cnet.batch_size]})

	###############
	# TRAIN MODEL #
	###############
	print '... training'
	# early-stopping parameters
	patience = 10000  # look as this many examples regardless
	patience_increase = 2  # wait this much longer when a new best is
						   # found
	improvement_threshold = 0.995  # a relative improvement of this much is
								   # considered significant
	validation_frequency = min(cnet.n_train_batches, patience / 2)
								  # go through this many
								  # minibatche before checking the network
								  # on the validation set; in this case we
								  # check every epoch

	best_params = None
	best_validation_loss = np.inf
	best_iter = 0
	test_score = 0.
	start_time = time.clock()

	epoch = 0
	done_looping = False

	while (epoch < n_epochs) and (not done_looping):
		epoch = epoch + 1
		for minibatch_index in xrange(cnet.n_train_batches):
			print 'minibatch index:', minibatch_index
			minibatch_avg_cost = train_model(minibatch_index)
			# iteration number
			iter = (epoch - 1) * cnet.n_train_batches + minibatch_index

			#if (iter + 1) % validation_frequency == 0:
			if iter % 5 == 0:
				# compute zero-one loss on validation set
				validation_losses = [validate_model(i) for i
									 in xrange(cnet.n_valid_batches)]
				this_validation_loss = np.mean(validation_losses)

				print('epoch %i, minibatch %i/%i, validation error %f %%' %
					 (epoch, minibatch_index + 1, cnet.n_train_batches,
					  this_validation_loss * 100.))

				# if we got the best validation score until now
				if this_validation_loss < best_validation_loss:
					#improve patience if loss improvement is good enough
					if this_validation_loss < best_validation_loss *  \
						   improvement_threshold:
						patience = max(patience, iter * patience_increase)

					best_validation_loss = this_validation_loss
					best_iter = iter

					# test it on the test set
					test_losses = [test_model(i) for i
								   in xrange(cnet.n_test_batches)]
					test_score = np.mean(test_losses)

					print(('     epoch %i, minibatch %i/%i, test error of '
						   'best model %f %%') %
						  (epoch, minibatch_index + 1, cnet.n_train_batches,
						   test_score * 1.))

			if patience <= iter:
					done_looping = True
					break

	end_time = time.clock()
	print(('Optimization complete. Best validation score of %f %% '
		   'obtained at iteration %i, with test performance %f %%') %
		  (best_validation_loss * 100., best_iter + 1, test_score * 100.))
	print >> sys.stderr, ('The code for file ' +
						  os.path.split(__file__)[1] +
						  ' ran for %.2fm' % ((end_time - start_time) / 60.))



if __name__ == '__main__':
	train_test()




