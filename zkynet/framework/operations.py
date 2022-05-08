import numpy as np
from .computation_graph import Function, Variable

class Identity(Function):
    def __init__(self):
        super().__init__("Identity", inputs=(Variable("x"),))

    def call(self, x):
        return x.value

    def grad(self, var):
        pass

class Add(Function):
    def __init__(self):
        super().__init__("Add", inputs=(Variable("a"), Variable("b")))

    def call(self, a, b):
        return a.value + b.value


class Multiply(Function):
    def __init__(self):
        super().__init__("Multiply", inputs=(Variable("a"), Variable("b")))

    def call(self, a, b):
        return a.value * b.value


class Square(Function):
    def __init__(self):
        super().__init__("Square", inputs=(Variable("x"),))

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
