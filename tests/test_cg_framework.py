import os, sys
ABS_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ABS_PATH, '../'))

import zkynet.framework as zn

description="testing CG framework"

class MyTestModel1(zn.Function):
    """A rather simple function that represents:

    f(x,w) = (x+w)*x^2

    where x is an input and w is a parameter.
    """
    def __init__(self, w0=1):
        super().__init__(inputs=(cg.Variable("x"),),
                         params=(cg.Parameter("w", w0),))

    def call(self, x):
        a = op.add(x, self.param_node("w"))
        b = op.square(x)
        c = op.mult(a, b)
        return c

######## Integrity tests ######################
# two calls of the same function results in
# two different computational graphs even
# if the graph structure are the same & nodes
# have the same values.


######## Equality tests ########################
# Test: Equality of nodes. Two Node objects are equal if:
# - they have the same ID
# - they have the same value
#
# Two Node objects have the same ID if:
# - they belong to the same computational graph (i.e. _same_ function
#      call; note one function call corresponds to one computational graph)
# - they correspond to the same template (Input or Function)
def test_Node_equality():
    pass
