import numpy as np

import theano
import theano.tensor as T
import theano.ifelse as ifelse
import pdb

np.set_printoptions(precision=2, linewidth=200, suppress=True)

class ACR(object):
    def __init__(self, ac, image_size):
        self.ac = ac
        self.image_size = image_size
        self.template = self.ac.template
        self.template_size = T.shape(self.template)[0]

    def render_minibatch(self, iGeoArray):
        # pdb.set_trace()
        results, updates = theano.scan(self.render, sequences=[iGeoArray[0], iGeoArray[1]])
        # pdb.set_trace()
        return results

    def index_to_coords(self, i):
        return (T.floor(i / self.image_size), i % self.image_size)

    def render(self, geoPose, intensity):
        # geoPose = iGeoPose[0]
        # intensity = iGeoPose[1]
        # geoPose = iGeoPose
        results, updates = theano.scan(lambda i: self.output_value_at(geoPose, self.index_to_coords(i)),
                                       sequences=[T.arange(self.image_size*self.image_size)])
        return results.reshape([self.image_size, self.image_size]) * intensity

    def get_template_value(self, template_x, template_y):
        # use true_div to prevent rounding error
        x = T.sum(T.cast(template_x + T.true_div(self.template_size - 1, 2), 'int32'))
        y = T.sum(T.cast(template_y + T.true_div(self.template_size - 1, 2), 'int32'))

        within_x_bounds = T.eq(T.ge(x, 0), T.le(x, self.template_size - 1))
        within_y_bounds = T.eq(T.ge(y, 0), T.le(y, self.template_size - 1))

        x_min_opts = T.zeros(2)
        x_min_opts = T.set_subtensor(x_min_opts[0], x)
        x_min_opts = T.set_subtensor(x_min_opts[1], self.template_size - 1)
        x_min = T.min(x_min_opts)

        x_max_opts = T.zeros(2)
        x_max_opts = T.set_subtensor(x_max_opts[0], x_min)
        x_max_opts = T.set_subtensor(x_max_opts[1], 0)
        x_inside = T.cast(T.max(x_max_opts), 'int32')

        y_min_opts = T.zeros(2)
        y_min_opts = T.set_subtensor(y_min_opts[0], y)
        y_min_opts = T.set_subtensor(y_min_opts[1], self.template_size - 1)
        y_min = T.min(y_min_opts)

        y_max_opts = T.zeros(2)
        y_max_opts = T.set_subtensor(y_max_opts[0], y_min)
        y_max_opts = T.set_subtensor(y_max_opts[1], 0)
        y_inside = T.cast(T.max(y_max_opts), 'int32')


        # inside_x = T.max([T.min(np.array([x, self.template_size - 1])), 0])
        # inside_y = T.max([T.min(np.array([y, self.template_size - 1])), 0])
        return ifelse.ifelse(
                            T.eq(within_x_bounds + within_y_bounds, 2),
                            self.template[x_inside, y_inside],
                            T.constant(np.float32(0.0)))

    def get_interpolated_template_value(self, template_x, template_y):
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
               (self.get_template_value(x_low, y_low)    * (x_high - template_x) * (y_high - template_y) + \
                self.get_template_value(x_high, y_low)   * (template_x - x_low)  * (y_high - template_y) + \
                self.get_template_value(x_low, y_high)   * (x_high - template_x) * (template_y - y_low) + \
                self.get_template_value(x_high, y_high)  * (template_x - x_low)  * (template_y - y_low))

    def output_value_at(self, geoPose, coords):
        output_x = coords[0]
        output_y = coords[1]
        output_coords = theano.shared(np.float32(np.ones(3)))
        #doing this -1 hack because we are using inctensor as subtensor has bugs in GPU
        output_x = output_x - 1
        output_y = output_y - 1
        output_coords = T.inc_subtensor(output_coords[0], output_x)
        output_coords = T.inc_subtensor(output_coords[1], output_y)

        template_coords = T.dot(geoPose, output_coords)
        template_x = template_coords[0]
        template_y = template_coords[1]

        return self.get_interpolated_template_value(template_x, template_y)
        # return self.get_template_value(template_x, template_y)





















