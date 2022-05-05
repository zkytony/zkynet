from .cg import Function

# class Identity(Function):
#     def call()

class Add(Function):
    def __init__(self, a, b):
        super().__init__(inputs={"a": a, "b": b})

    def call(self, a, b):
        return a + b

    def grad(self):
        return


class Multiply(Function):
    def __init__(self, a, b):
        super().__init__(inputs={"a": a, "b": b})

    def call(self, a, b):
        return a * b
