# A basic feedforward neural network with
# tunable parameters, and the backprop
# training algorithm
from .function import Function
import numpy as np

class FCN(Function):
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
        This is the forward pass of the network.
        """
        outputs = []
        inp = x
        for i in range(len(self._layers)):
            outputs.append(self._layers[i](inp))
            inp = outputs[-1]
        self._meta["layer_outputs"] = outputs
        return outputs[-1]


    def layer(self, i):
        """
        Returns a layer given index;
        Note that the first layer (i=0) is closest
        to the input, while the last layer is the farthest.
        """
        return self._layers[i]

    def __str__(self):
        result = "FCN:\n"
        num_params = 0
        for i in range(len(self._layers)):
            result += f"** LAYER {i}: {self.layer(i)}\n"
            num_params += self.layer(i).num_params
        result += f"TOTAL PARAMS: {num_params}"
        return result

class LinearLayer(Function):
    """
    A linear layer outputs y = x^T W + b
    where x is (n,1), W is (n,m), and b is (m,1)
    """
    def __init__(self, weights, bias):
        """
        Args:
            weights (np.array): a matrix (n,m)
            bias (np.array): a vector (m,)
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

    @property
    def weights(self):
        return self._weights

    @property
    def num_params(self):
        return self._weights.shape[0]*self._weights.shape[1] + len(self._bias)

    def __str__(self):
        return f"Linear(weights=\n{self._weights}, bias=\n{self._bias})"


class Identity(Function):
    """Identity function: just returns itself"""
    def __call__(self, x):
        return x


class ReLU(Function):
    """rectified linear unit g(x) = max{0,x}"""
    def __call__(self, x):
        return np.maximum(x, 0)

class FCNLayer(Function):
    """An FCN layer is a linear layer passed through
    a non-linear function.

    It outputs: y = g(L(x)) where L(x) is the linear layer
    """
    def __init__(self, linear_layer, g_func):
        """
        Args:
            linear_layer (LinearLayer)
            g_func (Function)
        """
        self._linear_layer = linear_layer
        self._g_func = g_func

    def __call__(self, x):
        return self._g_func(self._linear_layer(x))

    @classmethod
    def build(cls, shape, g_func=ReLU, weights=None, bias=None):
        """constructs a FCNLayer where the weights are of given `shape` with random
        initial weights, between 0 and 1), unless `weights` is specified.
        """
        assert len(shape) == 2, "weight matrix must be 2D."
        if weights is None:
            weights = np.random.rand(*shape)
        if bias is None:
            bias = np.random.rand(shape[1],)
        return FCNLayer(LinearLayer(weights, bias), g_func())

    def __str__(self):
        return f"FCNLayer({self._linear_layer},\n   g_func={self._g_func.__class__.__name__})"

    @property
    def num_params(self):
        return self._linear_layer.num_params
