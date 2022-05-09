import os, sys
ABS_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ABS_PATH, '../'))

from zkynet.framework import cg, op
from zkynet.visual import plot_cg
import numpy as np
import time

description="testing forward construction for the computational graph framework"

class MyTestModel1(cg.Module):
    """A rather simple function that represents:

    f(x,w) = (x+w)*x^2

    where x is an input and w is a parameter.
    """
    def __init__(self, w0=1):
        super().__init__("model1",
                         inputs=(cg.Variable("x"),),
                         params=(cg.Parameter("w", w0),))

    def call(self, x):
        a = op.add(x, self.param_node("w"))
        b = op.square(x)
        c = op.mult(a, b)
        return c

def test_model1_forward():
    m = MyTestModel1()
    assert m._params["w"] == 1  # initial value
    assert m.param_val("w") == 1  # initial value

    # forward pass; constructs computation graph,
    # and stores gradients.
    x = 3
    _start_time = time.time()
    result = m(x)
    print("forward pass time: {}s".format(time.time() - _start_time))
    _start_time = time.time()
    y = (x+m.param_val("w")) * x**2
    print("default computation time: {}s".format(time.time() - _start_time))
    assert result.value == 36


def test_visualize_cg():
    m = MyTestModel1()
    assert m._params["w"] == 1  # initial value

    # forward pass; constructs computation graph,
    # and stores gradients.
    x = np.array([3])
    result = m(x)
    plot_cg(result.root)


def run():
    test_model1_forward()
    test_visualize_cg()

if __name__ == "__main__":
    run()
