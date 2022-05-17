"""
Test linear model
"""
import os, sys
ABS_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ABS_PATH, '../'))

from zkynet.models import fcn
import jax.numpy as jnp
import torch

description="test linear model"

def test_linear_operator():
    W0 = jnp.array([[1, -1, 5],
                    [0, 14, -9]], dtype=jnp.float32)
    b0 = jnp.array([1, 4, -1], dtype=jnp.float32)
    linfn = fcn.Linear(W0, b0)
    x = jnp.array([1, -12], dtype=jnp.float32)
    y = linfn(x)
    assert (y.value == jnp.array([[2, -165, 112]])).all()

    y.back()
    print(y.grad(linfn.param("W")))


def run():
    test_linear_operator()


if __name__ == "__main__":
    run()
