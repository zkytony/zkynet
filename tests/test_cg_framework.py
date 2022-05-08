import os, sys
ABS_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ABS_PATH, '../'))

from zkynet.framework import cg

description="testing CG framework"

class MyTestModel1(cg.Function):
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
def test_call_integrity():
    m = MyTestModel1()
    result1 = m(3)
    result2 = m(3)
    assert result1 != result2
    assert result1.id != result2.id


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

def run():
    test_call_integrity()

if __name__ == "__main__":
    run()
