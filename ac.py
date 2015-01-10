import numpy as np
import pdb
import theano
import theano.tensor as T
theano.config.exception_verbosity = 'high'

class AC(object):
    def __init__(self, rng, template_size=3, template=None, activation=T.tanh):
        """
        The AC units. They consist of only a bias, with no input term.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type template_size: int
        :param template_size: dimensionality of AC

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer. If `None`, output will be just template.
        """
        if template is None:
            template_values = np.float32(np.asarray(rng.uniform(0, 1, (template_size, template_size))))
            template = theano.shared(value=template_values, name='template', borrow=True)
        self.template = template

        self.output = (self.template if activation is None
                       else activation(self.template))
        self.params = [self.template]
