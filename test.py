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
  template = theano.shared(np.float32(np.array([[0.22, 0.44, 0.22],
                                                [0.66, 0.88, 0.66],
                                                [0.11, 0.33, 0.11]])), name='template')
  image_size = 10
  # transform = np.float32(np.array([[12,-65,-98,27,0.5,12,math.pi/7], [12,-65,-98,27,0.5,12,math.pi/7]]))
  transform = np.float32(np.random.rand(100, 7))
  # pdb.set_trace()

  ack = ac.AC(None, template=template, activation=None)
  acker = acr.ACR(ack, theano.shared(image_size, name='image_size'))
  intm.getINTMMatrix(2, None, theano.shared(transform))
  # iGeoPose = ([0][0],
  #             intm.getINTMMatrix(1, None, theano.shared(transform))[1][0])
  # render = acker.render(iGeoPose[0], iGeoPose[1])
  # print render.eval()

  # cost = T.sum(T.pow(T.reshape(render, [1, image_size * image_size])
  #                 - theano.shared(np.random.rand(1, image_size * image_size))
  #             , 2))
  # print T.grad(T.sum(render), acker.template).eval()
  pdb.set_trace()

















