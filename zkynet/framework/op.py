import numpy as np
from .cg import Function, Variable

class Identity(Function):
    def __init__(self):
        super().__init__(inputs=(Variable("x")))

    def call(self, x):
        return x.value


class Add(Function):
    def __init__(self):
        super().__init__(inputs=(Variable("a"), Variable("b")))

    def call(self, a, b):
        return a.value + b.value


class Multiply(Function):
    def __init__(self, a, b):
        super().__init__(inputs=(Variable("a"), Variable("b")))

    def call(self, a, b):
        return a.value * b.value


class Square(Function):
    def __init__(self):
        super().__init__(inputs=(Variable("x")))

    def call(self, x):
        return np.square(x.value)
