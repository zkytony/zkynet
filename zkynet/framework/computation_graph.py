"""
A framework to represent computational graph.
"""

########### The computation graph components ##########
class IDObject:
    """Object with an ID; Note that
    the IDObject can carry an underlying
    'referee' which is used for equality
    comparison and hashing."""
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
    """
    def __init__(self, value, children=None, parents=None):
        """
        Args:
            children (list): list of children nodes of this node
            parents (dict): maps from FunctionNode (parent) to a string that
                indicates the name of the input to the parent function that
                this node corresponds to.
        """
        if children is None:
            children = []
        self._children = children
        if parents is None:
            parents = {}
        self._parents = parents
        self._value = value
        super().__init__(self.__class__.__name__)

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
                parents_str += f"{parent._fun.name}:{parent_input_name};"
            parents_str += "]"
        return parents_str

    def __repr__(self):
        return str(self)


class InputNode(Node):
    """A leaf node in the computational graph"""
    def __init__(self, name, value, parents=None):
        """
        Args:
            name: name of the input
        """
        super().__init__(value, parents=parents)
        self.name = name


class FunctionNode(Node):
    """A non-leaf node in the computational graph"""
    def __init__(self, fun, value, children, parents=None):
        """
        Args:
            fun (Function): the Function this node subsumes.
            children (list): list of children nodes of this node;
                note that order matters; the order of the children
                should match the order of inputs when calling the
                underlying function 'fun'.
        """
        self._fun = fun
        super().__init__(value,
                         children=children,
                         parents=parents)

    def __str__(self):
        parents_str = self._get_parents_str()
        return f"{self.__class__.__name__}<{self._fun.name}>({self.value}){parents_str}"

    @property
    def function(self):
        return self._fun

    def grad(self):
        """computes the gradient of the function
        with respect to every input and """


# algorithms to process computational graphs
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
