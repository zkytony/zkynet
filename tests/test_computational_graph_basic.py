import os, sys
ABS_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ABS_PATH, '../'))

from zkynet.framework import cg
from zkynet.framework import op
import numpy as np

class MyTestModel1(cg.Function):
    """A rather simple function that represents:

    f(x,w) = (x+w)*x^2

    where x is an input and w is a parameter.
    """
    def __init__(self, w0=1):
        super().__init__(inputs=(cg.Variable("x"),),
                         params=(cg.Parameter("w", w0),))

    def call(self, x):
        import pdb; pdb.set_trace()
        a = op.add(x, self.param_node("w"))
        b = op.square(x)
        c = op.mult(a, b)
        return c


def test_model1():
    m = MyTestModel1()
    assert m._params["w"] == 1  # initial value

    # forward pass; constructs computation graph,
    # and stores gradients.
    x = np.array([3])
    result = m(x)
    assert result == (3+1)*3**2

    # obtain the gradient of the function output with
    # respect to inputs (these are numbers that are
    # basically gradient function with assigned inputs
    # and parameters).
    dmdw = m.grad("w"); assert dmdw == 3**2
    dmdx = m.grad("x"); assert dmdx == 3**2 + (3+1)*2*3

    # obtain the gradient function; these are functions
    # that you can later use. Note that they are also
    # of type Function.
    fun_dmdw = m.grad_fn("w")
    fun_dmdx = m.grad_fn("x")
    assert fun_dmdw(x=3, w=1) == dmdw
    assert fun_dmdx(x=3, w=1) == dmdx

if __name__ == "__main__":
    test_model1()
