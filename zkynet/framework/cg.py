"""
A framework to define functions with corresponding,
dynamically generated computational graph. Gradients
are computed using automatic differentiation.
"""

def _check_inputs_instantiated(inputs):
    return all(inputs[name].instantiated
               for name in inputs)


def _get_uninstantiated_inputs(inputs):
    return {name: inputs[name]
            for name in inputs
            if not inputs[name].instantiated}

########## Template objects ###########
class Function:
    """
    A Function is an abstract template that maps
    inputs (ordered) to an output subject to some
    internal parameters.
    """
    def __init__(self, inputs, params={}):
        """
        Args:
            inputs (tuple): a tuple of ordered inputs, each a Variable.
            params (dict): maps from parameter name to a Parameter
                or a Constant.
        """
        assert all(isinstance(inputs[name], Variable) for name in inputs),\
            f"all objects in 'inputs' must be of type Variable"
        assert all(isinstance(params[name], Parameter)\
                   or isinstance(params[name], Constant) for name in params),\
            f"all objects in 'params' must be of type Parameter or Constant"

        self._ordered_input_names = (inp.name for inp in inputs)
        self._inputs = {inp.name: inp for inp in inputs}
        self._params = params

    def param(self, name):
        if name not in self._params:
            raise ValueError(f"{name} is not a parameter.")
        return self._params[name]

    @property
    def inputs(self):
        return self._inputs

    def call(self, *inputs, **call_args):
        """Function to be overriden"""
        raise NotImplementedError

    def __call__(self, *input_vals, **call_args):
        """The function is called (forward-pass).
        A computational graph is dynamically created.
        The input_vals will be converted (if not already)
        to a Node object.

        Args:
            *input_vals: each input is the value of an input
                that defines this function. Order matters.
                This value is either just a value (e.g. numpy array),
                an InputNode, or a FunctionNode.
            **call_args: call-time configurations to pass down
                 to call.

        Returns:
            Value: an object that represents a node in the grounded
                computational graph.
        """
        input_nodes = {}  # input nodes to this FunctionNode.

        try:
            for i in range(len(self._ordered_input_names)):
                input_name = self._ordered_input_names[i]
                if not isinstance(input_vals[i], Node):
                    node = InputNode(input_vals[i])
                    input_nodes[input_name] = node
                else:
                    input_nodes[input_name] = input_vals[i]
        except IndexError:
            raise ValueError("When calling a function, all its inputs must be instantiated.")

        output_val = self.call(*assigned_inputs, **call_args)

        # Wrap the output value as a FunctionNode, and connect the graph.
        output_node = FunctionNode(self, output_val, input_nodes)
        for input_name in input_nodes:
            input_nodes[input_name].set_parent(output_node, input_name)
        return output_node


class Input:
    """An Input is an abstract template
    for an input to a function, but without
    a value."""
    def __init__(self, name, input_type):
        self.name = name
        self.input_type = input_type


class Variable(Input):
    """Input variable; you have no control over."""
    def __init__(self, name):
        super().__init__(name, "variable")


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


########### The computation graph components ##########
class Node:
    """Node in the computation graph, a DAG.
    A node can always be regarded as an instantiation of
    a particular Input to a Function. It carries a value.

    We distinguish two node types: InputNode and FunctionNode.
    Don't confuse InputNode with Input; The InputNode
    literally refers to a leaf node on the DAG, while Input
    is just a placeholder of input in a Function template.

    The InputNode is a leaf node, while the FunctionNode
    is not a leaf node. Both should be grounded with values.
    The value of the FunctionNode represents the output
    of the function under some InputNode instantiation."""
    def __init__(self, value, children={}, parent=None, parent_input_name=None):
        """
        Args:
            children (dict): maps from string name to Node
            parent (FunctionNode): the node of the function that
                this node is an input for.
            parent_input_name (str): the name of the input to the
                parent function that this node corresponds to.
        """
        self._children = children
        self._parent = parent
        self._parent_input_name = parent_input_name

    def isleaf(self):
        return len(self._children) == 0


class InputNode(Node):
    """A leaf node in the computational graph"""
    def __init__(self, value, parent=None, parent_input_name=None):
        super().__init__(value, parent=parent,
                         parent_input_name=paren_input_name)

    def set_parent(self, parent, parent_input_name):
        self._parent = parent
        self._parent_input_name = parent_input_name


class FunctionNode(Node):
    """A non-leaf node in the computational graph"""
    def __init__(self, fun, value, children, parent=None, parent_input_name=None):
        """
        Args:
            fun (Function): the Function this node subsumes.
        """
        self._fun = fun
        super().__init__(value,
                         children=children,
                         parent=parent,
                         parent_input_name=paren_input_name)
