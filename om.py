import numpy

import theano
import theano.tensor as T
theano.config.exception_verbosity = 'high'
import pdb

def om(ac_contributions):
    """
    Expects a 3D tensor with each ACR's output
    (i.e., intensity at each pixel location)
    as the highest-level index.
    """

    # element-wise exp(a_vector * 100)
    ac_contributions = T.exp(T.mul(ac_contributions, 100))

    # sum the corresponding elements of each contribution,
    # then element-wise log them and divide by 100
    return T.true_div(T.log(T.sum(ac_contributions, axis=0)), 100)

