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
    def __init__(self, inputs={}):
        """
        Args:
            inputs (dict): maps from input name (str) to an Input object.
        """
        assert self._check_inputs_valid(inputs),\
            f"inputs must be either Function or numpy array. Got: {inputs}"

        self._vars = {name:inputs[name] for name in inputs
                      if isinstance(inputs[name], Variable)}
        self._params = {name:inputs[name] for name in inputs
                        if isinstance(inputs[name], Parameter)}
        self._constants = {name:inputs[name] for name in inputs
                           if isinstance(inputs[name], Constant)}

    def _check_inputs_valid(self, inputs):
        return all(isinstance(inputs[name], Input)
                   for name in inputs)

    def call(self, **inputs):
        """Function to be overriden"""
        raise NotImplementedError

    def grad(self, inpt):
        """Returns a dictionary that maps function that can be called to compute
        the gradient to this function"""
        raise NotImplementedError

    def __call__(self, **inputs):
        """The function is called (forward-pass)"""
        pass

    @property
    def params(self):
        return self._params


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

    def __str__(self):
        return "{}({})".format(self.__class__.__name__, self.value)

    def __repr__(self):
        return str(self)


class Variable(Input):
    """Input variable; you have no control over."""
    def __init__(self):
        super().__init__("variable")


class Parameter(Input):
    """Model parameter; you HAVE control over."""
    def __init__(self, init_value=None):
        super().__init__("parameter")
        self._value = init_value

    def __hash__(self):
        return hash(self._value)

    def __eq__(self, other):
        if isinstance(other, Parameter):
            return self.value == other.value
        else:
            return self.value == other


class Constant(Input):
    """Its value should not change; could
    be used to specify configuration of a
    function (e.g. kernel size of convolution)"""
    def __init__(self, val):
        super().__init__("constant")
        self._value = val

    def assign(self, v):
        raise ValueError("Constant value cannot change")
