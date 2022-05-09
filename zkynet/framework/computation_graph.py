"""
A framework to define functions with corresponding,
dynamically generated computational graph. Gradients
are computed using automatic differentiation.
"""
from .. import utils

########## Auxiliary objects ##########
class FunctionCallManager:
    """
    In order to enforce independence between computational
    graphs from different calls, FunctionCallManager will
    maintain the call ID of the current call, which is assigned
    to all nodes that are created during the call.

    It will clear the call ID if the call to the trigger
    function is finished (the trigger function is the first
    function that is called, which is likely a user-defined
    model).
    """
    def __init__(self):
        self.call_id = None
        self.trigger_function = None

    def call_begin(self, fun):
        if self.call_id is None:
            # We have no call now - so 'fun' is the trigger function
            self.call_id = utils.unique_id()
            self.trigger_function = fun
        else:
            # we only need to track whether the
            # trigger function has finished
            pass

    def call_end(self, fun):
        if self.call_id is None:
            raise ValueError("No call is happening.")
        if self.trigger_function.name != fun.name:
            # we only need to track whether the
            # trigger function has finished
            pass
        else:
            # OK. The trigger function has terminated.
            # So, we are done.
            self.call_id = None
            self.trigger_function = None
_GLOBAL_CALL_MANAGER = FunctionCallManager()


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
        self._inputs = {}
        for inp in inputs:
            inp.fun = self
            self._inputs[inp.name] = inp

        if params is None:
            params = set()
        assert all(isinstance(param, Parameter)\
                   or isinstance(param, Constant) for param in params),\
                   f"all objects in 'params' must be of type Parameter or Constant"
        self._params = {}
        for param in params:
            param.fun = self
            self._params[param.name] = param

    @property
    def name(self):
        return self._name

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
        """input nodes to this FunctionNode.
        Note: assume self._current_call_id is assigned"""
        input_nodes = []
        try:
            for i in range(len(self._ordered_input_names)):
                input_val = input_vals[i]
                input_name = self._ordered_input_names[i]
                if not isinstance(input_val, Node):
                    node = InputNode(_GLOBAL_CALL_MANAGER.call_id,
                                     self.inputs[input_name],
                                     input_val)
                    input_nodes.append(node)
                else:
                    input_nodes.append(input_val)
        except IndexError:
            raise ValueError("When calling a function, all its inputs must be instantiated.")
        return input_nodes

    def param_node(self, name):
        """Get an InputNode for the given parameter;
        Used to construct computational graph."""
        if name not in self._params:
            raise ValueError(f"{name} is not a parameter.")
        param = self._params[name]
        return InputNode(_GLOBAL_CALL_MANAGER.call_id, param, param.value)

    def param_val(self, n):
        return self._params[n].value

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
        _GLOBAL_CALL_MANAGER.call_begin(self)

        input_nodes = self._construct_input_nodes(*input_vals)
        output_val = self.call(*input_nodes, **call_args)

        # Wrap the output value as a FunctionNode, and connect the graph.
        if isinstance(output_val, FunctionNode):
            # "call" returns likely the output of running "call" for some other
            # function.  We extract its value, yet need to preserve the computation
            # graph, i.e. output_val will be the child node.
            output_node = FunctionNode(_GLOBAL_CALL_MANAGER.call_id, self,
                                       output_val.value, [output_val])
            output_val.add_parent(output_node, "preserve")
        else:
            output_node = FunctionNode(_GLOBAL_CALL_MANAGER.call_id, self,
                                       output_val, input_nodes)
            for i in range(len(input_nodes)):
                input_nodes[i].add_parent(output_node, self.input_name(i))

        _GLOBAL_CALL_MANAGER.call_end(self)
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
        if self._fun is None:
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


########### The computation graph components ##########
class IDObject:
    """Object with an ID; Two objects
    are the same if they have the same id."""
    def __init__(self, _id):
        self._id = _id

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
    Since it is a DAG, a node can have multiple children
    and multiple parents.

    We distinguish two node types: InputNode and FunctionNode.
    Don't confuse InputNode with Input; The InputNode
    literally refers to a leaf node on the DAG, while Input
    is just a placeholder of input in a Function template.

    The InputNode is a leaf node, while the FunctionNode
    is not a leaf node. Both should be grounded with values.
    The value of the FunctionNode represents the output
    of the function under some InputNode instantiation.

    Note: the notion of 'child' and 'parent' might be
    reversed to some people. Here, we mean:

       child --> parent

    because I want to view the input as the child and
    the output as the parent (that feels more natural)

    Note on equality:

       Two Node objects are equal if:
       - they have the same ID
       - they have the same value

       Two Node objects have the same ID if:
       - they belong to the same computational graph (i.e. _same_ function
            call; note one function call corresponds to one computational graph)
       - they instantiate the same object (Input or Function),
            i.e. they have the same 'ref' (identified by the 'functional name')
    """
    def __init__(self, call_id, ref, value, children=None, parents=None):
        """
        Args:
            call_id (str): the ID of the function call for which this
                node (or computational graph) is constructed. Note that
                all nodes on the same graph should have the same call id.
            ref (Function or Input): a reference to a Function or
                an Input object that this Node instantiates for.
            children (list): list of children nodes of this node
            parents (dict): maps from FunctionNode (parent) to a string that
                indicates the name of the input to the parent function that
                this node corresponds to.
        """
        self._call_id = call_id
        self._ref = ref
        if children is None:
            children = []
        self._children = children
        if parents is None:
            parents = {}
        self._parents = parents
        self._value = value
        _id = f"{self.__class__.__name__}_{call_id}_{ref.functional_name}"
        super().__init__(_id)

    def __hash__(self):
        return hash(self._id)

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.id == other.id\
                and self.value == other.value
        return False

    @property
    def ref(self):
        """reference to TemplateObject this node instantiates for."""
        return self._ref

    @property
    def value(self):
        return self._value

    def isleaf(self):
        return len(self._children) == 0

    @property
    def parents(self):
        return self._parents

    @property
    def parent_input_name(self, parent):
        return self._parents[parent]

    @property
    def children(self):
        return self._children

    def add_parent(self, parent, parent_input_name):
        self._parents[parent] = parent_input_name

    def __str__(self):
        parents_str = self._get_parents_str()
        return f"{self.__class__.__name__}({self.value}){parents_str}"

    def _get_parents_str(self):
        parents_str = ""
        if len(self._parents) > 0:
            parents_str = "-->["
            for parent, parent_input_name in self._parents.items():
                parents_str += f"{parent.function.name}:{parent_input_name};"
            parents_str += "]"
        return parents_str

    def __repr__(self):
        return str(self)


class InputNode(Node):
    """A leaf node in the computational graph"""
    def __init__(self, call_id, inpt, value, parents=None):
        assert isinstance(inpt, Input)
        super().__init__(call_id, inpt, value, parents=parents)

    @property
    def input(self):
        return self._ref


class FunctionNode(Node):
    """A non-leaf node in the computational graph"""
    def __init__(self, call_id, fun, value, children, parents=None):
        """
        Args:
            fun (Function): the Function this node subsumes.
            children (list): list of children nodes of this node;
                note that order matters; the order of the children
                should match the order of inputs when calling the
                underlying function 'fun'.
        """
        assert isinstance(fun, Function)
        super().__init__(call_id, fun, value,
                         children=children,
                         parents=parents)

    def __str__(self):
        parents_str = self._get_parents_str()
        return f"{self.__class__.__name__}<{self.function.name}>({self.value}){parents_str}"

    @property
    def function(self):
        return self._ref

    def grad(self):
        """computes the gradient of the function
        with respect to every input and """



########## algorithms to process computational graphs ##########
def get_input_nodes(root):
    """
    Args:
        root (Node): the root on a (sub) computational graph
    Output:
        set: a set of nodes of type InputNode
    """
    worklist = [root]
    seen = {root}
    input_nodes = set()
    while len(worklist) > 0:
        node = worklist.pop()
        if isinstance(node, InputNode):
            input_nodes.add(node)
        for child in node.children:
            if child not in seen:
                worklist.append(child)
                seen.add(child)
    return input_nodes

def get_all_nodes(root):
    """
    Args:
        root (Node): the root on a (sub) computational graph
    Output:
        set: a set of nodes of type InputNode
    """
    worklist = [root]
    seen = {root}
    all_nodes = set()
    while len(worklist) > 0:
        node = worklist.pop()
        all_nodes.add(node)
        for child in node.children:
            if child not in seen:
                worklist.append(child)
                seen.add(child)
    return all_nodes
