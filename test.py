import numpy as np
import matplotlib.pyplot as plt
import theano
# By convention, the tensor submodule is loaded as T
import theano.tensor as T
import theano.ifelse as ifelse

template = theano.shared(np.array([[0.3, 0.1], [0.7, 0.9]]))

def update(output_x, output_y):
  x_offset = 4
  y_offset = 4
  template_x = output_x - x_offset
  template_y = output_y - y_offset

  switch = ifelse.ifelse(T.eq(
                        T.eq(T.ge(template_x, 0), T.lt(template_x, 2)) +
                        T.eq(T.eq(T.ge(template_y, 0), T.lt(template_y, 2)), 1),
                    2),
                    template[template_x, template_y],
                    np.float64(0.0))
  return switch

def update_with_pose(output_x, output_y):
  geoPose = theano.shared(np.array([[ 2. ,  0. , -0. ],
                                    [ 0. ,  2. , -0.5],
                                    [ 0. ,  0. ,  1. ]]))
  template_coords = geoPose * np.array([x, y, 1])
  x_template = template_coords[0]
  y_template = template_coords[1]

  switch = ifelse.ifelse(T.eq(
                        T.eq(T.ge(template_x, 0), T.lt(template_x, 2)) +
                        T.eq(T.eq(T.ge(template_y, 0), T.lt(template_y, 2)), 1),
                    2),
                    template[template_x, template_y],
                    np.float64(0.0))
  return switch


def index_to_coords(i, side_length):
  return (T.floor(i / side_length), i % side_length)

results, updates = theano.scan(lambda i: apply(update_with_pose, index_to_coords(i, 10)),
                               sequences=[T.arange(10*10)])

results = results.reshape([10, 10])
print results.eval()