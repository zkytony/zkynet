import os, sys
ABS_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ABS_PATH, '../'))

from zkynet.framework import cg, op
import numpy as np

description="testing backprop gradient calculations for the computational graph framework"

class MyTestModel1(cg.Module):
    """A rather simple function that represents:

    f(x,w) = (x+w)*x^2

    where x is an input and w is a parameter.
    """
    def __init__(self, w0=1):
        super().__init__(inputs=(cg.Variable("x"),),
                         params=(cg.Parameter("w", w0),))

    def call(self, x):
        a = op.add(x, self.param("w"))
        b = op.square(x)
        c = op.mult(a, b)
        return c


def test_add_operator_gradient():
    add_op = op.Add()
    # a + b; dfda = 1
    dfda_fn = add_op.gradfn(cg.Variable("a"))
    assert dfda_fn(1,2).value == 1

def test_model1_gradient():
    m = MyTestModel1()
    result = m(3)
    result.back()

def run():
    test_add_operator_gradient()
    # test_model1_gradient()

if __name__ == "__main__":
    run()
