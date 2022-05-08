import os, sys
ABS_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ABS_PATH, '../'))

from zkynet.framework import cg, op

description="testing CG framework"

class MyTestModel1(cg.Function):
    """A rather simple function that represents:

    f(x,w) = (x+w)*x^2

    where x is an input and w is a parameter.
    """
    def __init__(self, w0=1):
        super().__init__("mymodel1",
                         inputs=(cg.Variable("x"),),
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

    # Make sure all nodes in one graph have the
    # same call id
    all_nodes1 = cg.get_all_nodes(result1)
    for n1 in all_nodes1:
        for n2 in all_nodes1:
            assert n1._call_id == n2._call_id


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
