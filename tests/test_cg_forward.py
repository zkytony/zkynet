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
        super().__init__(inputs=(cg.Variable("x"),),
                         params=(cg.Parameter("w", w0),))

    def call(self, x):
        a = op.add(x, self.param("w"))
        b = op.square(x)
        c = op.mult(a, b)
        return c

class CompositeModel_NoWeightSharing_DifferentInputs(cg.Module):
    """used to test composition"""
    def __init__(self):
        super().__init__(inputs=(cg.Variable("x1"),
                                 cg.Variable("x2")))
        # I expect the weights in the two may differ
        self._m1 = MyTestModel1()
        self._m2 = MyTestModel1()

    def call(self, x1, x2):
        a = self._m1(x1)
        b = self._m2(x2)
        return op.add(a, b)


def test_model1_forward():
    m = MyTestModel1()
    assert m.param("w").value == 1  # initial value

    x = 3
    _start_time = time.time()
    result = m(x)
    print("forward pass time: {}s".format(time.time() - _start_time))
    _start_time = time.time()
    y = (x+m.param("w").value) * x**2
    print("default computation time: {}s".format(time.time() - _start_time))
    assert result.value == 36


def test_visualize_cg():
    m = MyTestModel1()
    assert m._params["w"] == 1  # initial value
    x = np.array([3])
    result = m(x)
    plot_cg(result.root)


def test_visualize_CompositeModel_NoWeightSharing_DifferentInputs():
    m = CompositeModel_NoWeightSharing_DifferentInputs()
    result = m(3, 4)
    plot_cg(result.root, quiet=True)


def run():
    test_model1_forward()
    # test_visualize_cg()
    test_visualize_CompositeModel_NoWeightSharing_DifferentInputs()

if __name__ == "__main__":
    run()
