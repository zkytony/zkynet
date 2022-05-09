"""
A framework to define functions with corresponding,
dynamically generated computational graph. Gradients
are computed using automatic differentiation.
"""
from .. import utils
from dataclasses import dataclass

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
            # we only need to track hwether the
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

    @property
    def name(self):
        """
        The string that identifies the VARAIBLE name (or ENTITY)
        that this template object represents. For example,
        we could have two Function objects that represent
        the same function but we care about their outputs
        as separate variables. Then, these two Function objects
        should have the same 'functional_name' but different
        'name'.

        Assumption: 'name' should not contain any dot '.' character.
        """
        raise NotImplementedError


class Function(TemplateObject):
    """
    A Function is an abstract template that maps
    inputs (ordered) to an output subject to some
    internal parameters; the values of these parameters
    are kept tracked of in the model.
    """
    def __init__(self, inputs, params=None):
        """
        Args:
            inputs (tuple): a tuple of ordered inputs, each a Variable.
            params (list/set-like): parameters, either a Parameter or a Constant.
                order does not matter
        """
        self._functional_name = utils.fullname(self)
        self._name = "{}{}".format(self._functional_name, utils.unique_id(length=3))
        assert all(isinstance(inpt, Variable) for inpt in inputs),\
            f"all objects in 'inputs' must be of type Variable"
        self._ordered_input_names = tuple(inp.short_name for inp in inputs)
        self._inputs = {}
        for inp in inputs:
            inp.fun = self
            self._inputs[inp.short_name] = inp

        if params is None:
            params = set()
        assert all(isinstance(param, Parameter)\
                   or isinstance(param, Constant) for param in params),\
                   f"all objects in 'params' must be of type Parameter or Constant"
        self._params = {}
        for param in params:
            param.fun = self
            self._params[param.short_name] = param

    @property
    def name(self):
        """The variable name of a function should be
        thought of as the variable that represents
        the function's output. Two Functions that
        represent two different variables have different
        names; But they share the same "functional name"
        """
        return self._name

    @property
    def functional_name(self):
        """Function name without id"""
        return self._functional_name

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
           a OperatorNode, a number or array-like."""
        raise NotImplementedError

    def _construct_input_nodes(self, *input_vals):
        """input nodes to this Function.

        This is used at any step when buidling an
        OperatorNode during the computational graph
        instantiation. Each node in the output list
        correspond to both:
         (1) a slot in the inputs that define this function.
         (2) an element in 'input_vals'

        Note that an element in 'input_vals' could be:
        - a number or array
        - a Parameter or a Constant
        - an InputNode or OperatorNode
        """
        input_nodes = []
        try:
            for i in range(len(self._ordered_input_names)):
                input_val = input_vals[i]
                if isinstance(input_val, Node):
                    node = input_val
                elif isinstance(input_val, Input):
                    node = input_val.to_node()
                else:
                    # input_val is likely a number or array
                    input_name = self._ordered_input_names[i]
                    node = InputNode(_GLOBAL_CALL_MANAGER.call_id,
                                     self.inputs[input_name],
                                     input_val)
                input_nodes.append(node)
        except IndexError:
            raise ValueError("When calling a function, all its inputs must be instantiated.")
        return input_nodes

    def param(self, short_name):
        """Returns the Parameter corresponding to
        the given short_name, used when initializing
        this Function."""
        if short_name not in self._params:
            raise ValueError(f"{name} is not a parameter.")
        return self._params[short_name]

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
                a Parameter or Constant, an InputNode, or a OperatorNode.
            **call_args: call-time configurations to pass down
                 to call.

        Returns:
            OperatorNode: an object that represents a non-leaf node
                in the grounded computational graph.
            or a ModuleGraph that wraps a OperatorNode; the ModuleGraph
                is for the trigger function, which is what the user called.
                (This means, from a user's perspective, always expect getting
                 back a ModuleGraph when you call a Module.)
        """
        # The implementation is specific to Operator and Module. See below.
        raise NotImplementedError


class Operator(Function):
    """
    An operator is a function that we intend to
    hard code its derivatives. For such functions,
    we expect that the output of "call" should
    be a number, or array-like, instead of a
    OperatorNode. That means, implementation
    of an Operator's forward pass does not depend
    or use other operators. (Note that __call__
    still, as defined, returns a Node object.)
    """
    def __call__(self, *input_vals, **call_args):
        _GLOBAL_CALL_MANAGER.call_begin(self)

        input_nodes = self._construct_input_nodes(*input_vals)
        output_val = self.call(*input_nodes, **call_args)
        if isinstance(output_val, Node):
            raise ValueError("The output of Operator's 'call' function must not be a Node")
        output_node = OperatorNode(_GLOBAL_CALL_MANAGER.call_id, self,
                                   output_val, input_nodes)
        for i in range(len(input_nodes)):
            input_nodes[i].add_parent(output_node, self.input_name(i))

        _GLOBAL_CALL_MANAGER.call_end(self)
        return output_node


class Module(Function):
    """
    A Module is a function that is intended to
    be user-defined, complex functions whose
    forward call consists of operators and other
    modules. The gradient of this function is
    automatically computed using autodiff.

    There is a flat grounded computational graph
    corresponding to a module that is created when
    the module is called.
    """
    def __call__(self, *input_vals, **call_args):
        """
        If this Module is the trigger function, then
        we will return a ModuleGraph. Otherwise,
        return a OperatorNode.
        """
        _GLOBAL_CALL_MANAGER.call_begin(self)

        input_nodes = self._construct_input_nodes(*input_vals)
        output_val = self.call(*input_nodes, **call_args)

        if isinstance(output_val, OperatorNode):
            if _GLOBAL_CALL_MANAGER.trigger_function.name == self.name:
                # this is the trigger function;
                output = ModuleGraph(_GLOBAL_CALL_MANAGER.call_id,
                                   self, output_val)
            else:
                # so we just return output_val (OperatorNode)
                output = output_val
        else:
            output = OperatorNode(_GLOBAL_CALL_MANAGER.call_id, self,
                                  output_val, input_nodes)
            for i in range(len(input_nodes)):
                input_nodes[i].add_parent(output_node, self.input_name(i))

        _GLOBAL_CALL_MANAGER.call_end(self)
        return output


class Input(TemplateObject):
    """An Input is an abstract template
    for an input to a function, but without
    a value."""
    def __init__(self, short_name, input_type):
        """
        Args:
            short_name (str): short name of this input (e.g. 'x'),
                should indicate its role in the function
                that uses it. Note that the actual variable
                name of this Input will be prefixed (i.e. namespaced)
                by the function's variable name.
            input_type (str): identifies the type of input,
                for example 'variable' means it's based on
                observations, and 'parameter' means it is
                a function's self-maintained value.
        """
        self._short_name = short_name
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
    def name(self):
        """
        Name that represents the variable after being assigned to a function
        """
        if self._fun is None:
            raise ValueError("Input's function is NOT set. No variable to name.")
        return f"{self._fun.name}.{self._short_name}"

    @property
    def short_name(self):
        """The short name given at construction that does not
        uniquely identify this Input, but does identify this
        input with respect to its Function."""
        return self._short_name

    @property
    def functional_name(self):
        """
        The functional name that identifies both the function and the role
        this input plays to that function.
        """
        if self._fun is None:
            raise ValueError("Input's function is NOT set. No functional name.")
        return f"{self._fun.functional_name}.{self._short_name}"

    def to_node(self):
        """convert to InputNode; assuming this is called during a
        function call (so there would be a call id assigned to this
        InputNode)"""
        return InputNode(_GLOBAL_CALL_MANAGER.call_id,
                         self,
                         self.value)


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

    We distinguish two node types: InputNode and OperatorNode.
    Don't confuse InputNode with Input; The InputNode
    literally refers to a leaf node on the DAG, while Input
    is just a placeholder of input in a Function template.

    The InputNode is a leaf node, while the OperatorNode
    is not a leaf node. Both should be grounded with values.
    The value of the OperatorNode represents the output
    of the function (specifically, an Operator) under
    some InputNode instantiation.

    Note: the notion of 'child' and 'parent' might be
    reversed to some people. Here, we mean:

       child --> parent

    because I want to view the input as the child and
    the output as the parent (that feels more natural)

    *Note on equality:*

     * Two Node objects are equal if:
       (1) they have the same ID
       (2) they have the same value

     * Two Node objects have the same ID if:

       (1) they belong to the same computational graph
       (i.e. _same_ function call; note one function call
       corresponds to one computational graph)

       (2) they instantiate the same object (Input or Function),
       i.e. they have the same 'ref' (identified by the
       'functional name')


    *Note on restricting non-leaf nodes to OperatorNodes:*

        Our idea of an Operator is a Function with a hard-coded
        differentiation function. This makes up the "flat"
        computational graph, which is all that we need to properly
        compute gradients.

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
            parents (dict): maps from OperatorNode (parent) to a string that
                indicates the name of the input to the parent function that
                this node corresponds to.
        """
        _id = f"{self.__class__.__name__}_{call_id}_{ref.name}"
        super().__init__(_id)

        self.call_id = call_id
        self._ref = ref
        if children is None:
            children = []
        self._children = children
        if parents is None:
            parents = {}
        self._parents = parents
        self._value = value


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
                parents_str += f"{parent.operator.name}:{parent_input_name};"
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


class OperatorNode(Node):
    """A non-leaf node in the computational graph"""
    def __init__(self, call_id, op, value, children, parents=None):
        """
        Args:
            op (Operator): the Operator this node subsumes.
            children (list): list of children nodes of this node;
                note that order matters; the order of the children
                should match the order of inputs when calling the
                underlying function 'fun'.
        """
        assert isinstance(op, Function)
        super().__init__(call_id, op, value,
                         children=children,
                         parents=parents)

    def __str__(self):
        parents_str = self._get_parents_str()
        return f"{self.__class__.__name__}<{self.operator.name}>({self.value}){parents_str}"

    @property
    def operator(self):
        return self._ref

    def grad(self):
        """computes the gradient of the function
        with respect to every input and """

@dataclass(frozen=True)
class ModuleGraph:
    """
    A ModuleGraph is a computational graph that
    is grounded when a Module is called. It stores
    a flat computational graph (by 'flat' we mean
    that its internal OperatorNodes should only be
    Operators.)

    Note that since a Module's call may involve
    calling another module, we don't actually
    create a graph for that module. We only care
    about the trigger function (i.e. the first Module),
    similar to FunctionCallManager.

    Since the actual graph is captured by a Node,
    this class is very simplistic, but it does serve
    an important use.

    This object is assumed to be immutable
    """
    call_id: str

    # the trigger function
    module: Module

    # the root of the DAG
    # that represents the flat computational
    # graph for module instantiated with given
    # input values.
    root: OperatorNode

    @property
    def value(self):
        return self.root.value

    def __hash__(self):
        return hash(self.root.id)

    def __eq__(self, other):
        if isinstance(other, ModuleGraph):
            return self.call_id == other.call_id\
                and self.module.name == other.module.name\
                and self.root == other.root
        return False


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
