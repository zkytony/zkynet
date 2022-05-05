from .cg import Function, Variable

# class Identity(Function):
#     def call()

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
