"""
A framework to define functions with corresponding,
dynamically generated computational graph. Gradients
are computed using automatic differentiation.
"""
from .. import utils
from .computation_graph import FunctionNode, InputNode

########## Template objects ###########
class TemplateObject:
    @property
    def functional_name(self):
        """
        The name that identifies the ROLE this template
        object plays in the definition of a function; For example,
        if self is an Input, then this is the name that identifies
        both the function and the role this input plays to that function.
        """
        raise NotImplementedError


class Function(TemplateObject):
    """
    A Function is an abstract template that maps
    inputs (ordered) to an output subject to some
    internal parameters; the values of these parameters
    are kept tracked of in the model.
    """
    def __init__(self, name, inputs, params=None):
        """
        Args:
            inputs (tuple): a tuple of ordered inputs, each a Variable.
            params (list/set-like): parameters, either a Parameter or a Constant.
                order does not matter
        """
        self._name = name
        assert all(isinstance(inpt, Variable) for inpt in inputs),\
            f"all objects in 'inputs' must be of type Variable"
        self._ordered_input_names = tuple(inp.name for inp in inputs)
        self._inputs = {inp.name: inp for inp in inputs}

        if params is None:
            params = set()
        assert all(isinstance(param, Parameter)\
                   or isinstance(param, Constant) for param in params),\
                   f"all objects in 'params' must be of type Parameter or Constant"
        self._params = {param.name: param for param in params}

    def param_node(self, name):
        """Get an InputNode for the given parameter;
        Used to construct computational graph."""
        if name not in self._params:
            raise ValueError(f"{name} is not a parameter.")
        param = self._params[name]
        return InputNode(name, param.value)

    @property
    def functional_name(self):
        """
        The function's name
        """
        return self.name

    @property
    def inputs(self):
        return self._inputs

    def input_name(self, i):
        return self._ordered_input_names[i]

    def call(self, *input_nodes, **call_args):
        """Function to be overriden

        Args:
           *input_nodes (Node): nodes with values
                that are inputs to this function
                on the computation graph.
        Output:
           a FunctionNode, a number or array-like."""
        raise NotImplementedError

    def _construct_input_nodes(self, *input_vals):
        """input nodes to this FunctionNode."""
        input_nodes = []
        try:
            for i in range(len(self._ordered_input_names)):
                input_val = input_vals[i]
                if not isinstance(input_val, Node):
                    node = InputNode(self.input_name(i), input_val)
                    input_nodes.append(node)
                else:
                    input_nodes.append(input_val)
        except IndexError:
            raise ValueError("When calling a function, all its inputs must be instantiated.")
        return input_nodes


    def __call__(self, *input_vals, **call_args):
        """The function is called (forward-pass).
        A computational graph is dynamically created.
        The input_vals will be converted (if not already)
        to a Node object.

        Note: we enforce that two calls of the same function
        results in two different computational graphs even
        if the graph structure are the same & nodes
        have the same values.

        Args:
            *input_vals: each input is the value of an input
                that defines this function. Order matters.
                This value is either just a value (e.g. numpy array),
                an InputNode, or a FunctionNode.
            **call_args: call-time configurations to pass down
                 to call.

        Returns:
            FunctionNode: an object that represents a non-leaf node
                in the grounded computational graph.
        """
        _call_id = "{}-call{}".format(type(self), utils.unique_id())
        input_nodes = self._construct_input_nodes(call_id, *input_vals)
        output_val = self.call(*input_nodes, **call_args)

        # Wrap the output value as a FunctionNode, and connect the graph.
        if isinstance(output_val, FunctionNode):
            # "call" returns likely the output of running "call" for some other
            # function.  We extract its value, yet need to preserve the computation
            # graph, i.e. output_val will be the child node.
            output_node = FunctionNode(self, output_val.value, [output_val])
            output_val.add_parent(output_node, "preserve")
        else:
            output_node = FunctionNode(self, output_val, input_nodes)
            for i in range(len(input_nodes)):
                input_nodes[i].add_parent(output_node, self.input_name(i))
        return output_node


class Input(TemplateObject):
    """An Input is an abstract template
    for an input to a function, but without
    a value."""
    def __init__(self, name, input_type):
        """
        Args:
            name (str): name of this input (e.g. 'x'),
                should indicate its role in the function
                that uses it.
            input_type (str): identifies the type of input,
                for example 'variable' means it's based on
                observations, and 'parameter' means it is
                a function's self-maintained value.
        """
        self.name = name
        self.input_type = input_type
        self._value = None
        # Should be set upon the corresponding Function's __init__
        self._fun = None

    @property
    def value(self):
        return self._value

    def __str__(self):
        return f"{self.__class__.__name__}({self.name}, {self.value})"

    def __repr__(self):
        return str(self)

    @property
    def fun(self):
        """the function that this input is for"""
        return self._fun

    @fun.setter
    def fun(self, f):
        if self._fun is not None:
            raise ValueError("Input's function is already set.")
        if not isinstance(f, Function):
            raise TypeError("argument 'f' must be of type Function")
        self._fun = f

    @property
    def functional_name(self):
        """
        The name that identifies both the function and the role
        this input plays to that function.
        """
        if self._func is None:
            raise ValueError("Input's function is NOT set. No functional name.")
        return f"{self._fun.name}-{self.name}"


class Variable(Input):
    """Input variable; you have no control over.
    Nevertheless, this can be used to specifying how
    to validate an assignment to this variable at 'call'
    time (not yet implemented)."""
    def __init__(self, name):
        super().__init__(name, "variable")


class Parameter(Input):
    """Model parameter; you HAVE control over."""
    def __init__(self, name, init_value=None):
        super().__init__(name, "parameter")
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
    def __init__(self, name, val):
        super().__init__(name, "constant")
        self._value = val

    def assign(self, v):
        raise ValueError("Constant value cannot change")
