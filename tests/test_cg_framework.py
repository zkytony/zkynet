import os, sys
ABS_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ABS_PATH, '../'))

import copy
from zkynet.framework import cg, op


description="testing CG framework"

class MyTestModel1(cg.Module):
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
    assert isinstance(result1, cg.ModuleGraph)
    assert isinstance(result2, cg.ModuleGraph)

    # Make sure all nodes in one graph have the
    # same call id
    all_nodes1 = cg.get_all_nodes(result1.root)
    for n1 in all_nodes1:
        for n2 in all_nodes1:
            assert n1.call_id == n2.call_id
    all_nodes2 = cg.get_all_nodes(result2.root)
    for n1 in all_nodes2:
        for n2 in all_nodes2:
            assert n1.call_id == n2.call_id

    # Make sure nodes from different calls
    # have different call ids and different ids
    assert result1.call_id != result2.call_id
    assert result1.id != result2.id
    assert result1 != result2

######## Equality tests ########################
# Test: Equality of nodes. Two Node objects are equal if:
# - they have the same ID
# - they have the same value
#
# Two Node objects have the same ID if:
# - they belong to the same computational graph (i.e. _same_ function
#      call; note one function call corresponds to one computational graph)
# - they correspond to the same template (Input or Function)
def test_node_equality():
    m = MyTestModel1()
    result1 = m(3)
    result2 = m(3)
    all_nodes1 = cg.get_all_nodes(result1.root)
    all_nodes2 = cg.get_all_nodes(result2.root)
    # If we overwrite the ID of two nodes that
    # are otherwise the same, then we should get equality.
    for n1 in all_nodes1:
        if isinstance(n1, cg.FunctionNode):
            n1cp = n1.__class__(n1.call_id, n1._ref, n1.value,
                                children=n1.children, parents=n1.parents)
        else:
            n1cp = n1.__class__(n1.call_id, n1._ref, n1.value,
                                parents=n1.parents)
        assert n1cp == n1
        for n2 in all_nodes2:
            assert n1 != n2
            if type(n1) == type(n2)\
               and n1.value == n2.value:
                n2._id = n1._id
                assert n1 == n2

def run():
    test_call_integrity()
    test_node_equality()

if __name__ == "__main__":
    run()
