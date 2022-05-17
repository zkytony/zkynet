"""
A framework to define functions with corresponding,
dynamically generated computational graph. Gradients
are computed using automatic differentiation.

author: Kaiyu Zheng
"""
from .. import utils
from dataclasses import dataclass
import jax.numpy as jnp
from jax import jacrev, vjp, vmap

DEBUG=False

########## Auxiliary objects ##########
class CallSessionManager:
    """
    In order to enforce independence between computational
    graphs from different calls, CallSessionManager will
    maintain the call ID of the current call, which is assigned
    to all nodes that are created during the call.

    It will clear the call ID if the call to the trigger
    function is finished (the trigger function is the first
    function that is called, which is likely a user-defined
    model).

    Additionally, it stores InputNodes that have been
    created (identified by its ID), so that subsequent
    calls to the 'to_node' method of Input do not create
    new ones (which may have wrong parent/children relationships)
    but reuse the ones stored here.
    """
    def __init__(self):
        self.call_id = None
        self.trigger_function = None
        self._input_nodes_store = {}  # maps from ID to InputNode

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
            self._input_nodes_store = {}

    def get_input_node(self, node_id):
        return self._input_nodes_store.get(node_id, None)

    def store_input_node(self, node):
        if not isinstance(node, InputNode):
            raise TypeError("CallSessionManager only stores InputNode")
        if node.id in self._input_nodes_store:
            raise TypeError(f"Node {node.id} is already created. Unexpected.")
        self._input_nodes_store[node.id] = node
_GLOBAL_CALL_MANAGER = CallSessionManager()


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
    def __init__(self, inputs, params=None, functional_name=None):
        """
        Args:
            inputs (tuple): a tuple of ordered inputs, each a Variable.
            params (list/set-like): parameters, either a Parameter or a Constant.
                order does not matter
            functional_name (str): the canonical name of this Function.
                by default, it will be the import path of this class.
        """
        self._functional_name = functional_name
        if functional_name is None:
            self._functional_name = utils.fullname(self)
        self._name = "{}@{}".format(self._functional_name, utils.unique_id(length=3))
        assert all(isinstance(inpt, Variable) for inpt in inputs),\
            f"all objects in 'inputs' must be of type Variable"

        self._inputs = inputs  # will maintain the order
        self._inputs_dict = {}  # maps from input short name to input index
        for i, inp in enumerate(self._inputs):
            inp.fun = self
            self._inputs_dict[inp.short_name] = i

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

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    @property
    def inputs(self):
        return self._inputs

    @property
    def inputs_nofun(self):
        """Returns a tuple of Inputs aligned with self._inputs
        except that the 'fun' property of each is not assigned;
        That means those inputs could be used for other functions."""
        return tuple(inpt.copy_nofun() for inpt in self.inputs)

    def input_name(self, i):
        return self._inputs[i].name

    def input(self, short_name):
        if short_name not in self._inputs_dict:
            raise ValueError(f"{short_name} is not an input.")
        return self._inputs[self.input_index(short_name)]

    def input_index(self, short_name):
        if short_name not in self._inputs_dict:
            raise ValueError(f"{short_name} is not an input.")
        return self._inputs_dict[short_name]

    def call(self, *input_nodes):
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
            for i in range(len(self._inputs)):
                input_val = input_vals[i]
                if isinstance(input_val, Node):
                    node = input_val
                elif isinstance(input_val, Input):
                    node = _input_to_node(input_val)
                else:
                    # input_val is likely a number or array;
                    # it has a corresponding variable
                    variable_input = self._inputs[i]
                    assert isinstance(variable_input, Variable)
                    node = _input_to_node(variable_input,
                                          value=input_val)
                input_nodes.append(node)
        except IndexError:
            raise ValueError("When calling a function, all its inputs must be instantiated.")
        return input_nodes

    def param(self, short_name):
        """Returns the Parameter corresponding to
        the given short_name, used when initializing
        this Function."""
        if short_name not in self._params:
            raise ValueError(f"{short_name} is not a parameter.")
        return self._params[short_name]

    def __call__(self, *input_vals):
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

    def _verify_jax_array_type(self, *input_vals):
        """Ensure that we are only dealing with
        jax array"""
        for input_val in input_vals:
            if not isinstance(input_val, Node) and not isinstance(input_val, Input):
                if not isinstance(input_val, jnp.ndarray):
                    raise TypeError("Values to inputs must be JAX arrays")


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
    def __call__(self, *input_vals):
        _GLOBAL_CALL_MANAGER.call_begin(self)
        self._verify_jax_array_type(*input_vals)
        input_nodes = self._construct_input_nodes(*input_vals)
        output_val = self.call(*input_nodes)
        if isinstance(output_val, Node):
            raise ValueError("The output of Operator's 'call' function must NOT be a Node;"
                             "Must be a jax.ndarray.")
        output_node = OperatorNode(_GLOBAL_CALL_MANAGER.call_id, self,
                                   output_val, input_nodes)
        for i in range(len(input_nodes)):
            input_nodes[i].add_parent(output_node, i)

        _GLOBAL_CALL_MANAGER.call_end(self)
        return output_node

    def call(self, *input_nodes):
        """We will use JAX to implement the operator's logic. As a result,
        an Operator will have a _call function that takes in jax arrays
        as inputs (each is the value of the corresponding input node).
        """
        return self._op_impl(*(n.value for n in input_nodes))

    def _op_impl(self, *input_vals_ndarrays):
        raise NotImplementedError

    def _gradfn(self, inpt):
        """
        Returns the 'call_fun' used when building a Module,
        that represents the gradient of this operator with
        respect to input.  Using JAX.
        TO BE OVERRIDDEN.

        Args:
            inpt (Input): the gradient taken with respect for.
            *input_vals (list-like): the values to input when
                calling the gradient function.  Recall that
                df(a,b)/da could be written as df/da(a,b).
        Returns:
            a function that takes in *self.inputs
        """
        def _grad_call(*input_nodes):
            # This matches the signature of 'call' in Function
            inpt_i = self.input_index(inpt.short_name)
            return jacrev(self._op_impl, argnums=inpt_i)(*(n.value for n in input_nodes))
        return _grad_call

    def gradfn(self, inpt):
        """
        Returns the gradient function of this operator in
        the form of a Module. See *Module* for this use case.
        Suppose inpt represents variable v and this
        operator represents variable u, then this resulting
        gradient function is du/dv. **Recall that this gradient
        function takes the SAME inputs as this Operator.**

        Args:
            inpt (Input): the gradient taken with respect for.

        Returns:
            a Module that represents the gradient function
            of this operator with respect to inpt.
        """

        return Module.build(f"D{self.functional_name}#{inpt.short_name}",
                            self._gradfn(inpt),
                            self.inputs_nofun)

    def make_vjp(self, *input_vals_ndarrays):
        _, vjp_fun = vjp(self._op_impl, *input_vals_ndarrays)
        return vjp_fun



class Module(Function):
    """A Module is a function that is intended to be user-defined,
    (maybe) complicated functions whose forward call consists of
    operators and other modules. The gradient of this function is
    automatically computed using autodiff.

    There is a flat grounded computational graph corresponding to
    a module that is created when the module is called.

    One use case of Module is also when defining an Operator,
    you could define the gradient of that operator, potentially
    a composite function, as a Module, so that the computational
    graph represented by this gradient function STILL consists
    of the Operators.
    """
    def __init__(self, inputs, params=None, functional_name=None, call_fun=None):
        """
        Args:
            call_fun (function): function to be called. This
                supports dynamic construction of a new module
                with a custom call function. This function has
                the same definition as the 'call' in Function.
        """
        super().__init__(inputs, params=params, functional_name=functional_name)
        self._call_fun = call_fun

    def call(self, *input_nodes):
        if self._call_fun is None:
            raise NotImplementedError
        else:
            return self._call_fun(*input_nodes)

    def __call__(self, *input_vals):
        """
        If this Module is the trigger function, then
        we will return a ModuleGraph. Otherwise,
        return a OperatorNode.
        """
        _GLOBAL_CALL_MANAGER.call_begin(self)
        self._verify_jax_array_type(*input_vals)
        input_nodes = self._construct_input_nodes(*input_vals)
        output_val = self.call(*input_nodes)
        if isinstance(output_val, Node)\
           and not isinstance(output_val, OperatorNode):
            raise ValueError(f"If the 'call' function of {self.name} returns a Node,"\
                             "it must return an OperatorNode; Currently, it returns a"\
                             f"{type(output_val)}")

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
                input_nodes[i].add_parent(output, i)

        _GLOBAL_CALL_MANAGER.call_end(self)
        return output

    @classmethod
    def build(cls, functional_name, call_func, inputs, params=None):
        """
        Allows building a Module with a custom call function.

        Usage:
            >>> def mycall(x, y):
            >>>    return op.add(x,y)
            >>>
            >>> module = Module.build("myadd", mycall, (Variable("x"), Variable("y")))
            >>> result = module(1,3)
            >>> print(result.value)
            >>> # Output: 4
        """
        return Module(inputs,
                      params=params,
                      functional_name=functional_name,
                      call_fun=call_func)


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
        if self._fun is None:
            return f"{self.__class__.__name__}(NOFUNC.{self.short_name}, {self.value})"
        else:
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

    def copy_nofun(self):
        """Returns an Input that is a copy of self but without the 'fun'
        property set.
        """
        raise NotImplementedError


class Variable(Input):
    """Input variable; you have no control over.
    Nevertheless, this can be used to specifying how
    to validate an assignment to this variable at 'call'
    time (not yet implemented)."""
    def __init__(self, name):
        super().__init__(name, "variable")

    def copy_nofun(self):
        return Variable(self.short_name)


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

    def copy_nofun(self):
        return Parameter(self.short_name, init_value=self.value)


class Constant(Input):
    """Its value should not change; could
    be used to specify configuration of a
    function (e.g. kernel size of convolution)"""
    def __init__(self, name, val):
        super().__init__(name, "constant")
        self._value = val

    def assign(self, v):
        raise ValueError("Constant value cannot change")

    def copy_nofun(self):
        return Constant(self.short_name, self.value)


def _input_to_node(inpt, value=None):
    """convert 'inpt' (Input) to InputNode; assuming this is called during a
    function call (so there would be a call id assigned to this
    InputNode)

    Note that 'value' is used only if inpt is a Variable.
    Otherwise, the value comes from what is stored in the
    inpt object.
    """
    call_id = _GLOBAL_CALL_MANAGER.call_id
    input_node_id = InputNode.makeID(call_id, inpt)
    input_node = _GLOBAL_CALL_MANAGER.get_input_node(input_node_id)
    if input_node is not None:
        return input_node
    else:
        if isinstance(inpt, Variable):
            assert value is not None, "Variable input must ground with a value."
        else:
            value = inpt.value
        input_node = InputNode(call_id, inpt, value)
        _GLOBAL_CALL_MANAGER.store_input_node(input_node)
        return input_node


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
            parents (dict): maps from OperatorNode (parent) to an integer that
                indicates the index of the input to the parent function that
                this node corresponds to.
        """
        _id = self.__class__.makeID(call_id, ref)
        super().__init__(_id)

        if children is None:
            children = []
        if parents is None:
            parents = {}

        self.call_id = call_id
        self._ref = ref
        self._children = children
        self._parents = parents
        self._value = value
        # maps from parent to the partial gradient sent from the parent
        self._gradients_from_parents = {}
        self._gvalue = None

    @classmethod
    def makeID(cls, call_id, ref):
        return f"{cls.__name__}_{call_id}_{ref.name}"

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
        """forward pass value stored in this node"""
        return self._value

    @property
    def gvalue(self):
        """the gradient dF/dv where F is the trigger function and v is self"""
        return self._gvalue

    def isleaf(self):
        return len(self._children) == 0

    @property
    def parents(self):
        return self._parents

    def parent_input_index(self, parent):
        return self._parents[parent]

    @property
    def children(self):
        return self._children

    def add_parent(self, parent, parent_input_index):
        self._parents[parent] = parent_input_index

    ## Functions for backprop
    def receive(self, parent, gvalue):
        """receives gradient value from parent;
        This value represents dF/dv where v is the
        variable represented by 'self'"""
        if parent not in self._parents:
            raise ValueError(f"The node {parent} is not my parent.")
        self._gradients_from_parents[parent] = gvalue

    def received_messages_from_all_parents(self):
        return len(self._gradients_from_parents) == len(self._parents)

    def update_gvalue(self):
        """update gvalue to be the sum of gradients from
        parents. If there is no parent, imagine there is
        an OperatorNode that contains an identity function
        as the parent so the gradient is 1 (i.e. dF/dF = 1)"""
        if len(self._parents) == 0:
            self._gvalue = jacrev(lambda x: x, argnums=0)(self.value)
        else:
            self._gvalue = sum(self._gradients_from_parents[p]
                               for p in self._parents)

    def __str__(self):
        parents_str = self._get_parents_str()
        return f"{self.__class__.__name__}({self.value}){parents_str}"

    def _get_parents_str(self):
        parents_str = ""
        if len(self._parents) > 0:
            parents_str = "-->["
            for parent, parent_input_index in self._parents.items():
                parent_input_name = parent.ref.inputs[parent_input_index].short_name
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

    def send(self, child, gvalue):
        """
        sends dF/dc (where F is the trigger function) to the child
        """
        child.receive(self, gvalue)

    def grad(self, child):
        """computes the gradient of the function with respect to the
        child at 'child_index'.  Mathematically, suppose the
        operator node represents variable v, and the child
        represents variable c, then this computes dv/dc

        How the backwards computational graph works: Just like
        how `pytorch's computatinoal graph
        <https://pytorch.org/blog/computational-graphs-constructed-in-pytorch/>`_,
        works, the backward pass builds a DAG as well. However,
        here, **we do not (need to) represent that DAG
        explicitly**.  The gradients are abstracted as messages,
        and the input nodes to a gradient operator are never used
        again (Essentailly, we only build 'local' computational
        graphs for each gradient operator without connecting them
        together.

        In our framework, there is always only one flat grounded
        computational graph built in the forward pass. The
        gradient values, once 'back' is called by the Module, are
        stored within nodes on this graph.

        Args:
            child (Node): child node (could be leaf or non-leaf)

        Returns:
            number or array

        """
        # need the Input object for child's slot
        inpt = self.ref.inputs[child.parent_input_index(self)]
        gradfn = self.operator.gradfn(inpt)
        # Need to construct inputs to this function, in order to
        # compute its gradient value. Note that we want independence
        # between the forward graph and this gradient operator graph.
        input_vals = (ch.value for ch in self.children)
        return gradfn(*input_vals).value

    def vjp(self, child):
        """
        The Vector Jacobian Product. This is roughtly computing
        the chain rule from the function's output up to the child:

        dF/dp * dp/dc.

        Here, p = self, c = child, and F = Module. The dp/dc is
        the Jacobian matrix (in practice a tensor), while dF/dp is
        the "vector" (which in practice can be a tensor).
        """
        input_vals = (ch.value for ch in self.children)
        vjp_fun = self.operator.make_vjp(*input_vals)
        # We do vmap multiple times over vjp_fun to get a function that can
        # take in self.gvalue, a tensor.  Each application of vmap
        # increases one tensor dimension the resulting function can handle.
        # We know that vjp works for 1D tensor, vmap(vjp) works for 2D,
        # vmap(vmap(vjp)) works for 3D, etc.
        tensor_vjp_fun = vjp_fun
        # Number of tensor dimensions we shall vectorize the vjp function over;
        # It is the difference between the dimension vjp can handle (which is
        # based on the dimension of the output tensor of this operator, stored
        # in self.value) and the desired dimension we need (which is based on
        # self.gvalue, which can be seen as df/dp)
        vec_tensor_dims = len(self.gvalue.shape)-len(self.value.shape)
        if DEBUG:
            ss = "vmap(vjp_fun"
        for d in range(vec_tensor_dims):
            if DEBUG:
                ss = "vmap(" + ss
            tensor_vjp_fun = vmap(tensor_vjp_fun)
        if DEBUG:
            print(ss)
        return tensor_vjp_fun(self.gvalue)[child.parent_input_index(self)]


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
    similar to CallSessionManager.

    Since the actual graph is captured by a Node,
    this class is very simplistic, but it does serve
    an important use.

    This object is assumed to be immutable. So if
    you reassign its fields (e.g. call_id) then you
    are on your own.
    """
    def __init__(self, call_id, module, root):
        """
        Args:
            call_id (str): the ID of the call that this computational
                graph was built for.
            module (Module): the trigger function
            root (Node): the root of the DAG that represents the flat
                computational graph for module instantiated with given
                input values.
        """
        self.call_id = call_id
        self.module = module
        self.root = root
        self._input_nodes = {inp.id: inp
                             for inp in get_input_nodes(self.root)}

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

    def back(self):
        """
        Backpropagates gradients to every node in
        the computational graph. Mathematically,
        the gradient at each node is dF/dv where
        F is the variable at the root of the ModuleGraph,
        and v is the variable at the node.

        After this step, all nodes in the graph will
        have a "grad" property that stores dF/dv.
        """
        # This can be implemented through message passing over
        # the graph. At any time, there is (1) a number of
        # "senders", (2) a number of "receivers", (3) a number of
        # 'waiters', (4) a number of "done senders".  A sender
        # has received messages from all of its parents, so it
        # knows its gradient and starts converting its children
        # from 'waiters' into 'receivers,' if not already.  This
        # continues until all nodes are "done senders."  A
        # "sender" becomes a "done sender" if it has finished
        # passing a message to all its children.  Note that
        # 'waiters' refers to all nodes that we haven't reached
        # or ones that are children of 'receivers.'  Both 'waiter'
        # and 'done sender' are an abstract concept we don't need
        # to track.
        _senders = []
        _receivers = set({self.root})
        while not (len(_senders) == 0 and len(_receivers) == 0):
            # message passing
            while _senders:
                sender = _senders.pop()
                assert isinstance(sender, OperatorNode)
                for child in sender.children:
                    vjp = sender.vjp(child)
                    sender.send(child, vjp)
                    _receivers.add(child)
            # conversion from receiver to sender
            _still_receivers = set()
            for receiver in _receivers:
                if receiver.received_messages_from_all_parents():
                    receiver.update_gvalue()
                    if isinstance(receiver, OperatorNode):
                        _senders.append(receiver)
                else:
                    _still_receivers.add(receiver)
            _receivers = _still_receivers

    def grad(self, inpt):
        """
        Returns the gradient value with respect to the
        given input 'inpt'. Equivalent to fetching the InputNode
        and then return its 'gvalue' field. This is convenient
        after the user calls 'back' and gets a ModuleGraph, but
        the user doesn't directly have access to the InputNode
        objects inside this graph.

        Args:
            inpt (Input): the Input to the trigger function.
        """
        _id = InputNode.makeID(self.root.call_id, inpt)
        if _id not in self._input_nodes:
            raise ValueError(f"{inpt} is not an input (i.e. leaf node)"
                             "to this computational graph")
        return self._input_nodes[_id].gvalue


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
