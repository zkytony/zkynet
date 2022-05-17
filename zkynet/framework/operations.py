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

    def call(self, x):
        return self._call(x.value)

    def _call(self, x):
        return jnp.identity(x)

    def _gradfn(self, inpt):
        return jacrev(self._call, argnums=1)(x.value)


class Add(Operator):
    def __init__(self):
        super().__init__(inputs=(Variable("a"), Variable("b")))

    def call(self, a, b):
        return self._call(a.value, b.value)

    def _call(self, a, b):
        return a + b

    def _gradfn(self, inpt):
        def _a_grad(a, b):
            return jacrev(self._call, argnums=0)(a.value, b.value)
        def _b_grad(a, b):
            return jacrev(self._call, argnums=1)(a.value, b.value)
        if inpt.short_name == "a":
            return _a_grad
        elif inpt.short_name == "b":
            return _b_grad
        else:
            raise ValueError(
                f"Unknown input for {self.functional_name}: {inpt.short_name}")


class Multiply(Operator):
    def __init__(self):
        super().__init__(inputs=(Variable("a"), Variable("b")))

    def call(self, a, b):
        return self._call(a.value, b.value)

    def _call(self, a, b):
        return a * b  # element wise multiplication

    def _gradfn(self, inpt):
        def _a_grad(a, b):
            return jacrev(self._call, argnums=0)(a.value, b.value)
        def _b_grad(a, b):
            return jacrev(self._call, argnums=1)(a.value, b.value)
        if inpt.short_name == "a":
            return _a_grad
        elif inpt.short_name == "b":
            return _b_grad
        else:
            raise ValueError(
                f"Unknown input for {self.functional_name}: {inpt.short_name}")


class Square(Operator):
    def __init__(self):
        super().__init__(inputs=(Variable("x"),))

    def call(self, x):
        return self._call(x.value)

    def _call(self, x):
        return jnp.square(x)

    def _gradfn(self, inpt):
        def _x_grad(x):
            return jacrev(self._call, argnums=0)(x.value)
        if inpt.short_name == "x":
            return _x_grad
        else:
            raise ValueError(
                f"Unknown input for {self.functional_name}: {inpt.short_name}")


class Dot(Operator):
    def __init__(self):
        super().__init__(inputs=(Variable("a"), Variable("b")))

    def call(self, a, b):
        return self._call(a.value, b.value)

    def _call(self, a, b):
        return jnp.dot(a, b)

    def _gradfn(self, inpt):
        def _a_grad(a, b):
            return jacrev(self._call, argnums=0)(a.value, b.value)
        def _b_grad(a, b):
            return jacrev(self._call, argnums=1)(a.value, b.value)
        if inpt.short_name == "a":
            return _a_grad
        elif inpt.short_name == "b":
            return _b_grad
        else:
            raise ValueError(
                f"Unknown input for {self.functional_name}: {inpt.short_name}")


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
