# Try out jax's jacobian
import jax.numpy as jnp
from jax import jacrev

description = "test jax jacobians"

def f(z1, z2):
    return z1 * z2

def z1(x, w):
    return w*jnp.sin(jnp.square(x))

def z2(x, w):
    return x + w

def F(x, w):
    _z1 = z1(x,w)
    _z2 = z2(x,w)
    _f = f(_z1, _z2)
    return _f

def dot(t1, t2):
    # This is developed through trial and error;
    if len(t1.shape) <= 2 and len(t2.shape) <= 2:
        return jnp.dot(t1, t2)
    elif len(t1.shape) <= 4 and len(t2.shape) <= 4:
        return jnp.tensordot(t1, t2)
    else:
        # this seems to be the right call for tensor product
        # (though I am not sure why)
        return jnp.multiply(t1, t2)

def test_forward_backward(x, w):
    _z1 = z1(x,w)
    _z2 = z2(x,w)
    _f = f(_z1, _z2)

    print("Forward pass:")
    print(f"    _F({x}, {w}) =", _f)

    dFdz1 = jacrev(f, argnums=0)(_z1, _z2)
    dFdz2 = jacrev(f, argnums=1)(_z1, _z2)
    print("Backward pass:")
    print("    dF/dz1 =\n", dFdz1)
    print("    dF/dz2 =\n", dFdz2)

    dz2dw = jacrev(z2, argnums=1)(x, w)
    dz1dw = jacrev(z1, argnums=1)(x, w)
    dFdw = dot(dFdz1, dz1dw) + dot(dFdz2, dz2dw)
    print("    dF/dw  = dF/dz2 * dz2/dw =\n", dFdw)
    assert jnp.all(dFdw == jacrev(F, argnums=1)(x, w))
    dz1dx = jacrev(z1, argnums=0)(x, w)
    dz2dx = jacrev(z2, argnums=0)(x, w)
    dFdx = dot(dFdz1, dz1dx) + dot(dFdz2, dz2dx)
    print("    dF/dx  = dF/dz1 * dz1/dx =\n", dFdx)
    assert jnp.all(dFdx == jacrev(F, argnums=0)(x, w))

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
    x = jnp.array([[[[3., 4., 5.]],
                    [[-1., 2., -3.]]],
                   [[[2., 0., 5.]],
                    [[3., 1., -1.]]]])

    # w = jnp.array(1.)
    w = jnp.array([[[[1., 2., 1.]],
                    [[1., 1., 1.]]],
                   [[[-1., 1., 1.]],
                    [[1., 1., 1.]]]])
    test_forward_backward(x, w)

def run():
    test_scalar_input()
    test_vector_input()
    test_matrix_input()
    test_tensor_input()

if __name__ == "__main__":
    run()
