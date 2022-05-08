import os, sys
ABS_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ABS_PATH, '../'))

from zkynet.framework import cg

description="testing forward construction for the computational graph framework"

class MyTestModel1(cg.Function):
    """A rather simple function that represents:

    f(x,w) = (x+w)*x^2

    where x is an input and w is a parameter.
    """
    def __init__(self, w0=1):
        super().__init__(inputs=(cg.Variable("x"),),
                         params=(cg.Parameter("w", w0),))

    def call(self, x):
        a = op.add(x, self.param_node("w"))
        b = op.square(x)
        c = op.mult(a, b)
        return c

def test_model1_forward():
    m = MyTestModel1()
    assert m._params["w"] == 1  # initial value

    # forward pass; constructs computation graph,
    # and stores gradients.
    x = 3
    result = m(x)
    assert result.value == 36


def test_visualize_cg():
    m = MyTestModel1()
    assert m._params["w"] == 1  # initial value

    # forward pass; constructs computation graph,
    # and stores gradients.
    x = np.array([3])
    result = m(x)
    plot_cg(result)


def run():
    test_model1_forward()
    test_visualize_cg()

if __name__ == "__main__":
    run()
