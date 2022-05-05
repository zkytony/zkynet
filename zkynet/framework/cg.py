"""
Components for the computation graph (a DAG):

A node is either a Function or an Input.

An Input must be at the end of any path on the
graph. We can assign value to an Input.

A Function takes in multiple inputs. It
can be "called" to produce an output, given
that all inputs have assigned values.
"""

import numpy as np

class Function:
    def __init__(self, inputs=(), config={}):
        """
        Args:
            inputs (dict): the inputs to the function. This
                is a dict mapping from a name to the input. Each
                input should be either a primitive value (numpy
                array) or a Function object.
            config (dict): configurations (e.g. dimension);
                assumed to be immutable.
        """
        assert self._check_inputs_valid(inputs),\
            f"inputs must be either Function or numpy array. Got: {inputs}"
        self.inputs = inputs
        self.config = config

    def _check_inputs_valid(self, inputs):
        return all(isinstance(inp, Function) or isinstance(inp, np.ndarray)
                   for inp in inputs)

    def call(self):
        """Function to be overriden"""
        raise NotImplementedError

    def grad(self):
        """Returns a dictionary that maps function that can be called to compute
        the gradient to this function"""
        raise NotImplementedError

    def __call__(self):
        """The function is called (forward-pass)"""
        pass


class Input:
    def __init__(self, input_type):
        self.input_type = input_type
        self._value = None
        self._grad = None  # stores the gradient

    def assign(self, v):
        self._value = v

    @property
    def value(self):
        return self._value


class Variable(Input):
    def __init__(self):
        super().__init__("variable")


class Parameter(Input):
    def __init__(self, init_value=None):
        super().__init__("parameter")
        self._value = init_value
