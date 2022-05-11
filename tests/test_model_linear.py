"""
Test linear model
"""
import os, sys
ABS_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ABS_PATH, '../'))

from zkynet.models import fcn
import numpy as np
import torch

description="test linear model"

def test_linear_operator():
    W0 = np.array([[1, -1, 5],
                   [0, 14, -9]], dtype=np.float32)
    b0 = np.array([1, 4, -1], dtype=np.float32)
    linfn = fcn.Linear(W0, b0)
    x = np.array([[1, -12]], dtype=np.float32)
    y = linfn(x)
    assert (y.value == np.array([[2, -165, 112]])).all()

    # For gradient, we will test against pytorch;
    # btw: here is how you manually set weights for
    # a PyTorch module; But first, let's check forward.
    torch_linfn = torch.nn.Linear(2, 3, bias=True)
    torch_linfn.weight = torch.nn.Parameter(torch.tensor(W0.transpose()))
    torch_linfn.bias = torch.nn.Parameter(torch.tensor(b0))
    assert (torch_linfn(torch.tensor(x)) == torch.tensor(y.value)).all()

def run():
    test_linear_operator()


if __name__ == "__main__":
    run()
