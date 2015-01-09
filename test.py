import theano
import theano.tensor as T
import theano.ifelse as ifelse
import numpy as np
import pdb
import math

import intm
import AC
import ACR

if __name__ == '__main__':
  template = theano.shared(np.array([[0.22, 0.44, 0.22],
                                     [0.66, 0.88, 0.66],
                                     [0.11, 0.33, 0.11]]))

  ac = AC.AC(None, template=template, activation=None)
  acr = ACR.ACR(ac)
  iGeoPose = intm.getINTMMatrix(1, None, np.array([[1,-5,-5,0.5,0.5,0,math.pi/2]]))[0]
  print acr.render(iGeoPose).eval()


















