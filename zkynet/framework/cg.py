# cg: computation graph


class OpNode:
    """
    An OpNode is a node on the computational graph.
    It can have multiple inputs, and it can be called
    to produce some output. Each input is also an OpNode.
    Also, it stores a corresponding gradient function.
    """
    def __init__(self, inputs):
        self._inputs = inputs

    def call(self):
        raise NotImplementedError

    def grad(self):
        raise NotImplementedError


class InputNode(OpNode):
    def __init__(self):
        super.__init__()
