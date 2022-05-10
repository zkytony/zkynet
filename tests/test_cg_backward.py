
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

def test_multiply_operator_gradient():
    mult_op = op.Multiply()
    # a + b; dfda = 1
    dfda_fn = mult_op.gradfn(cg.Variable("a"))
    assert dfda_fn(1,2).value == 2
    dfdb_fn = mult_op.gradfn(cg.Variable("b"))
    assert dfdb_fn(1,2).value == 1

def test_square_operator_gradient():
    square_op = op.Square()
    # a + b; dfda = 1
    dfdx_fn = square_op.gradfn(cg.Variable("x"))
    assert dfdx_fn(2).value == 4

def test_node_grad_function():
    mult_op = op.Multiply()
    result = mult_op(3, 4)
    num = result.grad(result.children[0])
    assert num == 4

def test_model1_gradient():
    m = MyTestModel1()
    result = m(3)
    result.back()
    input_nodes = cg.get_input_nodes(result.root)
    for input_node in input_nodes:
        if input_node.ref.short_name == "x":
            # dF/dx = 6w+27 = 33
            assert input_node.gvalue == 33
        elif input_node.ref.short_name == "w":
            # dF/dw = x^2 = 9
            assert input_node.gvalue == 9
        else:
            raise ValueError(f"Unexpected input node to module {input_node}")

def test_model1_gradient_vectorized():
    m = MyTestModel1()
    result = m(np.array([3, 4, 5]))
    result.back()
    input_nodes = cg.get_input_nodes(result.root)
    for input_node in input_nodes:
        if input_node.ref.short_name == "x":
            # dF/dx = 6w+27 = 33
            assert np.all(input_node.gvalue == np.array([33, 56, 85]))
        elif input_node.ref.short_name == "w":
            # dF/dw = x^2 = 9
            assert np.all(input_node.gvalue == np.array([9, 16, 25]))
        else:
            raise ValueError(f"Unexpected input node to module {input_node}")

def run():
    test_add_operator_gradient()
    test_multiply_operator_gradient()
    test_square_operator_gradient()
    test_node_grad_function()
    test_model1_gradient()
    test_model1_gradient_vectorized()

if __name__ == "__main__":
    run()
