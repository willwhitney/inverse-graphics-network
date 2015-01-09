import numpy as np

import theano
import theano.tensor as T
import theano.ifelse as ifelse
import pdb

np.set_printoptions(precision=2, linewidth=200, suppress=True)

output_side_length = 10
def index_to_coords(i):
    return (T.floor(i / output_side_length), i % output_side_length)

class ACR(object):
    def __init__(self, ac):
        self.ac = ac
        self.template = self.ac.template
        self.template_size = T.shape(self.template)[0]

    def render_minibatch(self, iGeoArray):
        # pdb.set_trace()
        results, updates = theano.scan(self.render, sequences=[iGeoArray[0], iGeoArray[1]])
        # pdb.set_trace()
        return results

    def render(self, geoPose, intensity):
        # geoPose = iGeoPose[0]
        # intensity = iGeoPose[1]
        # geoPose = iGeoPose
        # pdb.set_trace()
        results, updates = theano.scan(lambda i: self.output_value_at(geoPose, index_to_coords(i)[0], index_to_coords(i)[1]),
                                       sequences=[T.arange(output_side_length*output_side_length)])
        return results.reshape([output_side_length, output_side_length]) * intensity

    def get_template_value(self, template_x, template_y):
        x = T.sum(T.cast(template_x + (self.template_size / 2), 'int64'))
        y = T.sum(T.cast(template_y + (self.template_size / 2), 'int64'))
        return ifelse.ifelse(
                            T.eq(
                                    T.eq(T.ge(x, 0), T.lt(x, self.template_size)) +
                                    T.eq(T.eq(T.ge(y, 0), T.lt(y, self.template_size)), 1),
                                2),
                            self.template[x, y],
                            T.constant(np.float64(0.0)))

    def get_interpolated_template_value(self, template_x, template_y):
        x_low = T.floor(template_x)
        x_high = ifelse.ifelse(T.eq(T.ceil(template_x), template_x),template_x + 1.0,T.ceil(template_x))

        y_low = T.floor(template_y)
        y_high = ifelse.ifelse(
                                T.eq(T.ceil(template_y), template_y),
                                template_y + 1.0,
                                T.ceil(template_y))

        return (1. / ((x_high - x_low) * (y_high - y_low))) * \
               (self.get_template_value(x_low, y_low)    * (x_high - template_x) * (y_high - template_y) + \
                self.get_template_value(x_high, y_low)   * (template_x - x_low)  * (y_high - template_y) + \
                self.get_template_value(x_low, y_high)   * (x_high - template_x) * (template_y - y_low) + \
                self.get_template_value(x_high, y_high)  * (template_x - x_low)  * (template_y - y_low))

    def output_value_at(self, geoPose, output_x, output_y):
        output_coords = theano.shared(np.ones(3))
        output_coords = T.set_subtensor(output_coords[0], output_x)
        output_coords = T.set_subtensor(output_coords[1], output_y)

        template_coords = T.dot(geoPose, output_coords)
        template_x = template_coords[0]
        template_y = template_coords[1]

        return self.get_interpolated_template_value(template_x, template_y)





















