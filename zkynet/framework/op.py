from .cg import Function, Variable

# class Identity(Function):
#     def call()

class Add(Function):
    def __init__(self):
        super().__init__(inputs={"a": Variable(), "b": Variable()})

    def call(self, **inputs):


        return a + b

    def grad(self):
        return


class Multiply(Function):
    def __init__(self, a, b):
        super().__init__(inputs={"a": a, "b": b})

    def call(self, a, b):
        return a * b
