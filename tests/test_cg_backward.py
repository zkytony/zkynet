
import os, sys
ABS_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ABS_PATH, '../'))

from zkynet.framework import cg, op
import numpy as np

from test_cg_forward import (MyTestModel1,
                             CompositeModel_NoWeightSharing_DifferentInputs,
                             CompositeModel_NoWeightSharing_SameInputs,
                             CompositeModel_WeightSharing_DifferentInputs,
                             CompositeModel_WeightSharing_SameInputs)

description="testing backprop gradient calculations for the computational graph framework"


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
    # dF/dw = x^2 = 9
    assert result.grad(m.param("w")) == 9
    # dF/dx = 6w+27 = 33
    assert result.grad(m.input("x")) == 33

def test_model1_gradient_vectorized():
    m = MyTestModel1()
    result = m(np.array([3, 4, 5]))
    result.back()
    assert np.all(result.grad(m.param("w")) == np.array([9, 16, 25]))
    assert np.all(result.grad(m.input("x")) == np.array([33, 56, 85]))

def test_composite_model_gradient():
    cm1 = CompositeModel_NoWeightSharing_DifferentInputs()
    result = cm1(3,4)
    result.back()
    assert result.grad(cm1._m1.param("w")) == 9
    assert result.grad(cm1._m2.param("w")) == 16
    assert result.grad(cm1.input("x1")) == 33

    cm2 = CompositeModel_NoWeightSharing_SameInputs()
    result = cm2(3)
    result.back()
    print("dM2/dw1", result.grad(cm2._m1.param("w")))
    print("dM2/dw2", result.grad(cm2._m2.param("w")))
    print("dM2/dx", result.grad(cm2.input("x1")))
    print("---")

    cm3 = CompositeModel_WeightSharing_DifferentInputs()
    result = cm3(3,4)
    result.back()
    print("dM3/dw", result.grad(cm3._m1.param("w")))
    print("dM3/dx1", result.grad(cm3.input("x1")))
    print("dM3/dx2", result.grad(cm3.input("x2")))
    print("---")

    cm4 = CompositeModel_WeightSharing_SameInputs()
    result = cm4(4)
    result.back()
    print("dM4/dw", result.grad(cm4._m1.param("w")))
    print("dM4/dx1", result.grad(cm4.input("x1")))
    print("---")


def run():
    test_add_operator_gradient()
    test_multiply_operator_gradient()
    test_square_operator_gradient()
    test_node_grad_function()
    test_model1_gradient()
    test_model1_gradient_vectorized()
    test_composite_model_gradient()

if __name__ == "__main__":
    run()
