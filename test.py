import theano
import theano.tensor as T
import theano.ifelse as ifelse
import numpy as np
import intm
import pdb
import math
### next:
# fix the reset vector
# reverse the order of matrix operations
np.set_printoptions(precision=2, linewidth=200)

template = theano.shared(np.array([[0.22, 0.44],
                                   [0.66, 0.88]]))

# rawGeoPose =  np.array([[1.,0.,-0.5], [0.,1.,-0.5], [0.,0.,1.]]) \
#             * np.array([[1.,0.,0.],    [0.,1.,1.],    [0.,0.,1.]]) \
#             * np.array([[1.,0.,0.],    [0.,1.,1.],    [0.,0.,1.]]) \
#             * np.array([[1.,0.,0.],    [0.,1.,1.],    [0.,0.,1.]]) \
#             * np.array([[1.,0.,0.],    [0.,1.,1.],    [0.,0.,1.]]) \
#             * np.array([[1.,0.,0.5],  [0.,1.,0.5],  [0.,0.,1.]])
# geoPose = theano.shared(rawGeoPose)
geoPose = intm.getINTMMatrix(1, None, np.array([[1,-6,-5,0.5,0.5,1,math.pi/2]]))[0]
print geoPose.eval()
# pdb.set_trace()
# geoPose = theano.shared(np.array([[ 0.5 ,  0.  , -0.  ],
#        [ 0.  ,  0.5 , 6.25],
#        [ 0.  ,  0.  ,  1.  ]]))

def update(output_x, output_y):
  x_offset = 4
  y_offset = 4
  template_x = output_x - x_offset
  template_y = output_y - y_offset

  return get_template_value(template_x, template_y)


def get_template_value(template_x, template_y):
  x = T.sum(T.cast(template_x, 'int64'))
  y = T.sum(T.cast(template_y, 'int64'))
  return ifelse.ifelse(
                      T.eq(
                          T.eq(T.ge(x, 0), T.lt(x, 2)) +
                          T.eq(T.eq(T.ge(y, 0), T.lt(y, 2)), 1),
                        2),
                      template[x, y],
                      np.float64(0.0))

def get_interpolated_template_value(template_x, template_y):
  x_low = T.floor(template_x)
  x_high = ifelse.ifelse(
              T.eq(T.ceil(template_x), template_x),
              template_x + 1.0,
              T.ceil(template_x))

  y_low = T.floor(template_y)
  y_high = ifelse.ifelse(
              T.eq(T.ceil(template_y), template_y),
              template_y + 1.0,
              T.ceil(template_y))

  return (1. / ((x_high - x_low) * (y_high - y_low))) * \
     (get_template_value(x_low, y_low)    * (x_high - template_x) * (y_high - template_y) + \
      get_template_value(x_high, y_low)   * (template_x - x_low)  * (y_high - template_y) + \
      get_template_value(x_low, y_high)   * (x_high - template_x) * (template_y - y_low) + \
      get_template_value(x_high, y_high)  * (template_x - x_low)  * (template_y - y_low))


def update_with_pose(output_x, output_y):
  output_coords = theano.shared(np.ones(3))
  output_coords = T.set_subtensor(output_coords[0], output_x)
  output_coords = T.set_subtensor(output_coords[1], output_y)

  template_coords = T.dot(geoPose, output_coords)
  template_x = template_coords[0]
  template_y = template_coords[1]

  return get_interpolated_template_value(template_x, template_y)


def index_to_coords(i, side_length):
  return (T.floor(i / side_length), i % side_length)

results, updates = theano.scan(lambda i: apply(update_with_pose, index_to_coords(i, 15)),
                               sequences=[T.arange(15*15)])

if __name__ == '__main__':
  results = results.reshape([15, 15])
  print results.eval()


















