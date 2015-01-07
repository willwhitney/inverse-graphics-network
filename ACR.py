import numpy

import theano
import theano.tensor as T

def update_amount(template, geoPose, x, y):
    template_coords = geoPose * np.array([x, y, 1])
    x_template = template_coords[0]
    y_template = template_coords[1]
    x_low = T.floor(x_template)
    x_high = T.ceil(x_template)
    y_low = T.floor(y_template)
    y_high = T.ceil(y_template)

    result = 1 / ((x_high - x_low) * (y_high - y_low)) * \
                (   template[x_low][y_low]    * (x_high - x) * (y_high - y) + \
                    template[x_high][y_low]   * (x - x_low)  * (y_high - y) + \
                    template[x_low][y_high]   * (x_high - x) * (y - y_low) + \
                    template[x_high][y_high]  * (x - x_low)  * (y - y_low))

    return result


class ACR(object):
    def __init__(self, ac):
        self.ac = ac

    def render(self, iGeoPose):
        geoPose = iGeoPose[0]
        intensity = iGeoPose[1]

        # output = theano.shared(value=np.zeros(28, 28), name='output', borrow=True)
        theano.scan(fn=lambda prior_output, x, y: T.inc_subtensor(prior_output[x][y], update_amount(template, geoPose,x,y)),
                    outputs_info=output,
                    sequences=[np.array(range(28), range(28))]
                    )


        template = self.ac.b


igp = np.matrix()