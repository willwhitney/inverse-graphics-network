import theano
import theano.tensor as T
import theano.ifelse as ifelse
import numpy as np
import pdb
import math

import intm
import ac
import acr

if __name__ == '__main__':
  template = theano.shared(np.array([[0.22, 0.44, 0.22],
                                     [0.66, 0.88, 0.66],
                                     [0.11, 0.33, 0.11]]))

  ack = ac.AC(None, template=template, activation=None)
  acker = acr.ACR(ack)
  iGeoPose = intm.getINTMMatrix(1, None, np.array([[1,-5,-5,0.5,0.5,0,math.pi/2]]))[0]
  print acker.render(iGeoPose).eval()


















