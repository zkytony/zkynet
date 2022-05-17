"""
Hard-coded operations

The operations here are the most basic building
blocks. For more complex operations, look into
specific models in zkynet.models
"""
import jax.numpy as jnp
from jax import jacrev
from .computation_graph import Operator, Variable, Module

class Identity(Operator):
    def __init__(self):
        super().__init__(inputs=(Variable("x"),))

    def _op_impl(self, x):
        return jnp.identity(x)


class Add(Operator):
    def __init__(self):
        super().__init__(inputs=(Variable("a"), Variable("b")))

    def _op_impl(self, a, b):
        return a + b


class Multiply(Operator):
    def __init__(self):
        super().__init__(inputs=(Variable("a"), Variable("b")))

    def _op_impl(self, a, b):
        return a * b  # element wise multiplication


class Square(Operator):
    def __init__(self):
        super().__init__(inputs=(Variable("x"),))

    def _op_impl(self, x):
        return jnp.square(x)


class Dot(Operator):
    def __init__(self):
        super().__init__(inputs=(Variable("a"), Variable("b")))

    def _op_impl(self, a, b):
        return jnp.dot(a, b)


def add(a, b):
    return Add()(a, b)

def mult(a, b):
    return Multiply()(a, b)

def square(x):
    return Square()(x)

def identity(x):
    return Identity()(x)

def dot(a, b):
    return Dot()(a, b)
