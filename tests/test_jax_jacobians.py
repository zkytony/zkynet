# Try out jax's jacobian
import jax.numpy as jnp
from jax import jacrev

description = "test jax jacobians"

def f(z1, z2):
    return z1 * z2

def z1(x):
    return jnp.square(x)

def z2(x, w):
    return x + w

def F(x, w):
    _z1 = z1(x)
    _z2 = z2(x,w)
    _f = f(z1, z2)
    return _f

def test_forward_backward(x, w):
    _z1 = z1(x)
    _z2 = z2(x,w)
    _f = f(z1, z2)

    print("Forward pass:")
    print("    _F({x}, {w}) =", _f)

    dFdz1 = jacrev(f, argnums=0)(_z1, _z2)
    dFdz2 = jacrev(f, argnums=1)(_z1, _z2)
    print("Backward pass:")
    print("    dF/dz1 =", dFdz1)
    print("    dF/dz2 =", dFdz2)

    dz2dw = jacrev(z2, argnums=1)(x, w)
    dFdw = dFdz2 * dz2dw
    print("    dF/dw  = dF/dz2 * dz2/dw =", dFdw)
    assert dFdw == jacrev(F, argnums=1)(x, w)
    dz1dx = jacrev(z1, argnums=0)(x)
    dz2dx = jacrev(z2, argnums=0)(x, w)
    dFdx = dFdz1 * dz1dx + dFdz2 * dz2dx
    print("    dF/dx  = dF/dz1 * dz1/dx =", dFdx)

def test_scalar_input():
    print("============= test_scalar_input ==============")
    x = jnp.array(3.)
    w = jnp.array(1.)
    test_forward_backward(x, w)

def test_vector_input():
    print("============= test_vector_input ==============")
    x = jnp.array([3., 4., 5.])
    w = jnp.array(1.)
    test_forward_backward(x, w)

def test_matrix_input():
    print("============= test_matrix_input ==============")
    x = jnp.array([[3., 4., 5.],
                   [-1., 2., -3.]])
    w = jnp.array(1.)
    test_forward_backward(x, w)

def test_tensor_input():
    print("============= test_tensor_input ==============")
    x = jnp.array([[[3., 4., 5.],
                    [-1., 2., -3.]],
                   [[2., 0., 5.],
                    [3., 1., -1.]]])
    w = jnp.array(1.)
    test_forward_backward(x, w)

def run():
    test_scalar_input()
    test_vector_input()
    test_matrix_input()
    test_tensor_input()

if __name__ == "__main__":
    run()
