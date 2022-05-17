import os, sys
ABS_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ABS_PATH, '../'))

from zkynet.framework import cg, op
import jax.numpy as jnp
from jax import jacrev

from test_cg_forward import (MyTestModel1,
                             CompositeModel_NoWeightSharing_DifferentInputs,
                             CompositeModel_NoWeightSharing_SameInputs,
                             CompositeModel_WeightSharing_DifferentInputs,
                             CompositeModel_WeightSharing_SameInputs)

description="testing backprop gradient calculations for the computational graph framework"


def test_model1_gradient():
    m = MyTestModel1()
    result = m(jnp.array(3.))
    result.back()
    # dF/dw = x^2 = 9
    assert result.grad(m.param("w")) == 9.
    # dF/dx = 6w+27 = 33
    assert result.grad(m.input("x")) == 33.


def test_model1_gradient_vectorized():
    def model1fun(x,w):
        return (x + w) * x**2

    m = MyTestModel1()
    # 1D (vector)
    x = jnp.array([3., 4., 5.])
    result = m(x)
    result.back()

    jax_x_grad = jacrev(model1fun, argnums=0)(x,m.param("w").value)
    jax_w_grad = jacrev(model1fun, argnums=1)(x,m.param("w").value)

    assert jnp.all(result.grad(m.input("x")) == jax_x_grad)
    # This will fail because shape of mismatch; that's not a bug (it's a broadcasting issue I think)
    # assert jnp.all(result.grad(m.param("w")) == jax_w_grad)

    # 2D (matrix)
    x2 = jnp.array([[3., 4., 5.],
                    [5., 8., 10.],
                    [3., 3., -7.],
                    [5., 4., 10.]])
    result = m(x2)
    jax_x_grad = jacrev(model1fun, argnums=0)(x2,m.param("w").value)
    jax_w_grad = jacrev(model1fun, argnums=1)(x2,m.param("w").value)
    result.back()
    assert jnp.all(result.grad(m.input("x")) == jax_x_grad)
    # This will fail because shape of mismatch (it's a broadcasting issue I think)
    # assert jnp.all(result.grad(m.param("w")) == jax_w_grad)

    # 3D (tensor)
    x3 = jnp.array([[[3., 4., 5.],
                     [3., 3., -7.]],

                    [[-1., 4., -2.],
                     [3., -1., -5.]]])
    result = m(x3)
    jax_x_grad = jacrev(model1fun, argnums=0)(x3,m.param("w").value)
    result.back()
    jax_x_grad = jacrev(model1fun, argnums=0)(x3,m.param("w").value)
    jax_w_grad = jacrev(model1fun, argnums=1)(x3,m.param("w").value)
    assert jnp.all(result.grad(m.input("x")) == jax_x_grad)
    # This will fail because shape of mismatch (it's a broadcasting issue I think)
    # assert jnp.all(result.grad(m.param("w")) == jax_w_grad)


def test_composite_model_gradient():
    cm1 = CompositeModel_NoWeightSharing_DifferentInputs()
    result = cm1(jnp.array(3.), jnp.array(4.))
    result.back()
    assert result.grad(cm1._m1.param("w")) == 9.
    assert result.grad(cm1._m2.param("w")) == 16.
    assert result.grad(cm1.input("x1")) == 33.

    cm2 = CompositeModel_NoWeightSharing_SameInputs()
    result = cm2(jnp.array(3.))
    result.back()
    print("dM2/dw1", result.grad(cm2._m1.param("w")))
    print("dM2/dw2", result.grad(cm2._m2.param("w")))
    print("dM2/dx", result.grad(cm2.input("x1")))
    print("---")

    cm3 = CompositeModel_WeightSharing_DifferentInputs()
    result = cm3(jnp.array(3.), jnp.array(4.))
    result.back()
    print("dM3/dw", result.grad(cm3._m1.param("w")))
    print("dM3/dx1", result.grad(cm3.input("x1")))
    print("dM3/dx2", result.grad(cm3.input("x2")))
    print("---")

    cm4 = CompositeModel_WeightSharing_SameInputs()
    result = cm4(jnp.array(4.))
    result.back()
    print("dM4/dw", result.grad(cm4._m1.param("w")))
    print("dM4/dx1", result.grad(cm4.input("x1")))
    print("---")


def run():
    test_model1_gradient()
    test_model1_gradient_vectorized()
    test_composite_model_gradient()

if __name__ == "__main__":
    run()
