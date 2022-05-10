"""
Hard-coded operations
"""
import numpy as np
from .computation_graph import Operator, Variable, Module

class Identity(Operator):
    def __init__(self):
        super().__init__(inputs=(Variable("x"),))

    def call(self, x):
        return x.value

    def grad(self, var):
        return 1


class Add(Operator):
    def __init__(self):
        super().__init__(inputs=(Variable("a"), Variable("b")))

    def call(self, a, b):
        return a.value + b.value

    def grad(self, inpt):
        def _a_grad(a, b):
            return a
        def _b_grad(a, b):
            return b
        if inpt.short_name == "a":
            return Module.build("DAdd#a", _a_grad, self.inputs)


class Multiply(Operator):
    def __init__(self):
        super().__init__(inputs=(Variable("a"), Variable("b")))

    def call(self, a, b):
        return a.value * b.value


class Square(Operator):
    def __init__(self):
        super().__init__(inputs=(Variable("x"),))

    def call(self, x):
        return np.square(x.value)


def add(a, b):
    return Add()(a, b)

def mult(a, b):
    return Multiply()(a, b)

def square(x):
    return Square()(x)

def identity(x):
    return Identity()(x)
