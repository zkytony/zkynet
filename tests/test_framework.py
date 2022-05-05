from zkynet.framework import cg
from zkynet.framework import op
import numpy as np

class MyTestModel1(Function):
    """A rather simple function that represents:

    f(x,w) = (x+w)*x^2

    where x is an input and w is a parameter.
    """
    def __init__(self, w0=1):
        super().__init__(inputs={"x": cg.Variable(),
                                 "w": cg.Parameter(w0)})

    def call(self, x):
        a = op.add(x, self.w)
        b = op.square(x)
        c = op.mult(a, b)
        return c

def test_model1():
    m = MyTestModel1()
    print(m.params)   # expect to print out 'w'
    x = np.array([3])
    # We have now taken a forward-pass on m.
    # it should accumulate gradients.
    assert m(x) == (3+1)*3**2
