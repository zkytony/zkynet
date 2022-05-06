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
    def __init__(self, inputs, params=None):
        """
        Args:
            inputs (tuple): a tuple of ordered inputs, each a Variable.
            params (list/set-like): parameters, either a Parameter or a Constant.
                order does not matter
        """
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
    def name(self):
        return self.__class__.__name__

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
        input_nodes = self._construct_input_nodes(*input_vals)
        output_val = self.call(*input_nodes, **call_args)

        # Wrap the output value as a FunctionNode, and connect the graph.
        if isinstance(output_val, FunctionNode):
            # "call" returns likely the output of running "call" for some other
            # function.  We extract its value, yet need to preserve the computation
            # graph, i.e. output_val will be the child node.
            output_node = FunctionNode(self, output_val.value, [output_val])
            output_val.set_parent(output_node, "preserve")
        else:
            output_node = FunctionNode(self, output_val, input_nodes)
            for i in range(len(input_nodes)):
                input_nodes[i].set_parent(output_node, self.input_name(i))
        return output_node


class Input:
    """An Input is an abstract template
    for an input to a function, but without
    a value."""
    def __init__(self, name, input_type):
        self.name = name
        self.input_type = input_type
        self._value = None

    @property
    def value(self):
        return self._value

    def __str__(self):
        return f"{self.__class__.__name__}({self.name}, {self.value})"

    def __repr__(self):
        return str(self)


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


########### The computation graph components ##########
class IDObject:
    """Object with an ID"""
    COUNTER = {}
    def __init__(self, objtype):
        if objtype not in IDObject.COUNTER:
            IDObject.COUNTER[objtype] = 0
        next_id = IDObject.COUNTER[objtype]
        self._id = f"{objtype}_{next_id}"
        IDObject.COUNTER[objtype] += 1

    def __hash__(self):
        return hash(self._id)

    def __eq__(self, other):
        if isinstance(other, IDObject):
            return self._id == other._id
        else:
            return False

    @property
    def id(self):
        return self._id


class Node(IDObject):
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
    def __init__(self, value, children=None, parent=None, parent_input_name=None):
        """
        Args:
            children (list): list of children nodes of this node
            parent (FunctionNode): the node of the function that
                this node is an input for.
            parent_input_name (str): the name of the input to the
                parent function that this node corresponds to.
        """
        if children is None:
            children = []
        self._children = children
        self._parent = parent
        self._parent_input_name = parent_input_name
        self._value = value
        super().__init__(self.__class__.__name__)

    @property
    def value(self):
        return self._value

    def isleaf(self):
        return len(self._children) == 0

    @property
    def parent(self):
        return self._parent

    @property
    def parent_input_name(self):
        return self._parent_input_name

    @property
    def children(self):
        return self._children

    def set_parent(self, parent, parent_input_name):
        self._parent = parent
        self._parent_input_name = parent_input_name

    def __str__(self):
        func_str = ""
        if self._parent is not None and self._parent_input_name is not None:
            func_str = f"-->{self._parent._fun.name}:{self._parent_input_name}"
        return f"{self.__class__.__name__}({self.value}){func_str}"

    def __repr__(self):
        return str(self)


class InputNode(Node):
    """A leaf node in the computational graph"""
    def __init__(self, name, value, parent=None, parent_input_name=None):
        """
        Args:
            name: name of the input
        """
        super().__init__(value, parent=parent,
                         parent_input_name=parent_input_name)
        self.name = name


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
                         parent_input_name=parent_input_name)

    def __str__(self):
        func_str = ""
        if self._parent is not None and self._parent_input_name is not None:
            func_str = f"-->{self._parent._fun.name}[{self._parent_input_name}]"
        return f"{self.__class__.__name__}<{self._fun.name}>({self.value}){func_str}"

    @property
    def function(self):
        return self._fun
