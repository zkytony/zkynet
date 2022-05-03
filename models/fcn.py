# A basic feedforward neural network with
# tunable parameters, and the backprop
# training algorithm
from . import framework as zkyn
import numpy as np

class FCN(zkyn.Function):
    """
    FCN: fully-connected networks, also called deep
    feedforward neural networks, or multilayer perceptrons.

    An FCN defines a mapping

        y = f(x;p)

    where x are the inputs, y are the outputs, and p are
    the parameters. Specifically,

        y = fn (... (f3 (f2 (f1 (x) ) )

    where each f(x) = g(Wx + b) is a non-linear function
    that takes as input a linear transformation.
    """

    def __init__(self, layers):
        """
        Args:
            layers (list): A list of FCNLayer objects,
                which represent layers in the FCN.
                Note that the output of the i-1(th) layer
                is the input to the i(th) layer.
        """
        super().__init__()
        self._layers = layers


    def __call__(self, x):
        """

        """
        outputs = []
        inp = x
        for layer in self._layers:
            outputs.append(layer(inp))
            inp = outputs[-1]
        self._meta["layer_outputs"] = outputs
        return outputs[-1]


    def layer(self, i):
        """
        Returns a layer given index;
        Note that the first layer (i=0) is closest
        to the input, while the last layer is the farthest.
        """
        return self.layers[i]


class LinearLayer(mynn.Function):
    """
    A linear layer outputs y = Wx + b
    """
    def __init__(self, weights, bias):
        """
        Args:
            weights (np.array): a matrix nxm
        """
        self._weights = weights
        self._bias = bias

    def __call__(self, x):
        """
        Args:
            x (np.array): input of shape (D, n); each
                data point is an n-element vector.
        """
        return np.dot(x, self._weights) + self._bias


class FCNLayer(mynn.Function):
    """An FCN layer is a linear layer passed through
    a non-linear function.

    It outputs: y = g(L(x)) where L(x) is the linear layer
    """
    def __init__(self, linear_layer, g_func):
        self._linear_layer = linear_layer
        self._g_func = linear_layer

    def __call__(self, x):
        return self._g_func(self._linear_layer(x))


class ReLU(mynn.Function):
    """rectified linear unit g(x) = max{0,x}"""
    def __call__(self, x):
        return np.max(x, 0)
