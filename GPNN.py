"""
Simple implementation of the encoder GPNN for MNIST.

Uses a single hidden layer with a tanh activation function.

Uses code from http://deeplearning.net/tutorial/mlp.html, which is included
in this repo as `mlp.py`. More should be cut from it to do training, etc.
"""


import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T
theano.config.exception_verbosity = 'high'

from logistic_sgd import LogisticRegression, load_data

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None
                       else activation(lin_output))
        # parameters of the model
        self.params = [self.W, self.b]


class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function thanh or the
    sigmoid function (defined here by a ``SigmoidalLayer`` class)  while the
    top layer is a softamx layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie
        """

        # Since we are dealing with a two hidden layer MLP, this will
        # translate into a TanhLayer connected to a TanhLayer connected
        # to the LogisticRegression layer
        self.hiddenLayers = []
        self.hiddenLayers.append(HiddenLayer(rng=rng, input=input,
                                       n_in=n_in, n_out=n_hidden,
                                       activation=T.tanh))
        # self.hiddenLayers.append(HiddenLayer(rng=rng,
        #                                input=self.hiddenLayers[0].output,
        #                                n_in=n_hidden, n_out=n_hidden,
        #                                activation=T.tanh))
        # self.hiddenLayers.append(HiddenLayer(rng=rng,
        #                                input=self.hiddenLayers[1].output,
        #                                n_in=n_hidden, n_out=n_hidden,
        #                                activation=identity))

        # self.outputLayer = LogisticRegression(
        #         input=self.hiddenLayers[-1].output,
        #         n_in=n_hidden,
        #         n_out=n_out)

        self.outputLayer = HiddenLayer(
                rng=rng,
                input=self.hiddenLayers[-1].output,
                n_in=n_hidden,
                n_out=n_out,
                activation=identity)

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = sum([abs(layer.W).sum() for layer in self.hiddenLayers]) \
                + abs(self.outputLayer.W).sum()

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = sum([(layer.W ** 2).sum() for layer in self.hiddenLayers]) \
                    + (self.outputLayer.W ** 2).sum()

        self.output = self.outputLayer.output

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = reduce(lambda l1,l2: l1 + l2.params,
                             self.hiddenLayers,
                             []) \
                         + self.outputLayer.params

    # for linear neurons
    @staticmethod
    def identity(x):
        return x

    @staticmethod
    def logistic(x):
        return 1 / (1 + T.exp(-x))



