import numpy

import theano
import theano.tensor as T

class ACR(object):
    def __init__(self, ac):
        self.ac = ac

    def render(self, iGeoPose):
        template = self.ac.b
