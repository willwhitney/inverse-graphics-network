import numpy

import theano
import theano.tensor as T
theano.config.exception_verbosity = 'high'

class AC(object):
    def __init__(self, rng, n_values=11*11, template=None, activation=T.tanh):
        """
        The AC units. They consist of only a bias, with no input term.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type n_values: int
        :param n_values: dimensionality of AC

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer. If `None`, output will be just template.
        """

        if template is None:
            b_values = np.asarray(rng.randint(0, 2, n_values))
            template = theano.shared(value=b_values, name='template', borrow=True)
        self.template = template

        self.output = (self.template if activation is None
                       else activation(self.template))
        self.params = [self.template]
