import theano
import theano.tensor as T
theano.config.exception_verbosity = 'high'

import GPNN
import intm
import AC
import numpy as np
import pdb

class CapsuleNetwork(object):
	def __init__(self):
		self.num_acrs = 10 #number of ACRs
		self.output = None
		self.loss = None
		self.params = None
		self.rng = np.random.RandomState(123)
		# self.theano_rng = RandomStreams(rng.randint(2 ** 30))

		if False:
			datasets = load_data(dataset)
			self.train_set_x, self.train_set_y = datasets[0]

			# compute number of minibatches for training, validation and testing
			self.n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
		else:
			self.n_train_batches = 10

		# allocate symbolic variables for the data
		self.index = T.lscalar()    # index to a [mini]batch
		#self.x = T.matrix('x')  # the data is presented as rasterized images	
		self.x = theano.shared(np.random.rand(self.n_train_batches,28*28))

	def combineCapsules(ACs):
		for i in range(len(ACs)):
			AC = ACs[i]

		return image

	def create_model(self):
		self.encoder = GPNN.GPNN(rng=self.rng, input=self.x, n_in=28 * 28, n_hidden=2000, n_out=self.num_acrs*7)

		self.iGeoArray = dict()
		self.ACArray = dict()
		for i in range(self.num_acrs):
			igeon_indx = range(i,i+7) #pose + intensity
			self.iGeoArray[i] = intm.getINTMMatrix(self.n_train_batches,self.rng, self.encoder.output[:,igeon_indx])
			pdb.set_trace()
			# self.ACArray[i] = AC(self.rng)
		#combine capsule ACRs
		# rendering = self.combineCapsules(self.ACArray)


if __name__ == '__main__':
	net = CapsuleNetwork()
	net.create_model()
	# net.run()


