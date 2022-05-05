"""
Components for the computation graph (a DAG):

A node is either a Function or an Input.

An Input must be at the end of any path on the
graph. We can assign value to an Input.

A Function takes in multiple inputs (i.e. nodes). It
can be "called" to produce an output value, given
that all inputs have assigned values.

There is no special meaning for edges other
than connectivity.
"""

import numpy as np

class Node:
    """Node in the computation graph, a DAG"""
    def __init__(self, children):
        """
        Args:
            children (dict): mapping from name to Node
            parents (dict): mapping from name to Node
        """
        self._children = children

    @property
    def isleaf(self):
        return len(self._children) == 0


class Function(Node):
    """
    A Function takes inputs (each of type Input) and
    produces an output (of type Value). The output
    is only defined when all input variables are assigned.
    """
    def __init__(self, inputs={}):
        """
        Args:
            inputs (dict): maps from input name (str) to an Input object.
        """
        assert self._check_inputs_valid(inputs),\
            f"inputs must be either Function or numpy array. Got: {inputs}"
        super().__init__(inputs)
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
        """The function is called (forward-pass)

        Args:
            **inputs: mapping from input name to value.
                This will assign the inputs to this function
                to the given value. Note that (1) only Variables
                and Parameters can be assigned (the latter should
                be optional); (2) the function throws an exception
                if not all Variables are assigned a value.
        Returns:
            Value: an object

        """
        pass

    @property
    def params(self):
        return self._params


class Input(Node):
    """An Input is a leaf node on the DAG"""
    def __init__(self, input_type):
        super().__init__({})
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

    @property
    def instantiated(self):
        """Returns true if the variable is instantiated."""
        return self._value is not None


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


class Value(Node):
    """Think about it as a function but after it is called, so
    it is "grounded" to the input values. The inputs given
    will be all converted to constants (separate from the
    abstract computation graph).

    Note: Value is immutable, and it is NOT designed to
    be an input to a Function."""
    def __init__(self, inputs, val):
        """
        Args:
           inputs (dict): maps from name to Input; should be instantiated.
           val (array-like): the actual value
        """
        assert self._check_inputs_instantiated(inputs),\
            "The inputs to a Value node must have been instantiated."

        # convert inputs to constants, since they are no longer changeable.
        _inputs = {name: Constant(inputs[name].value)}

        super().__init__(_inputs)
        self._value = val

    def _check_inputs_instantiated(self, inputs):
        return all(inputs[name].instantiated
                   for name in inputs)
