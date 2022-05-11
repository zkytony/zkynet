"""
Test linear model
"""
import os, sys
ABS_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ABS_PATH, '../'))

from zkynet.models import fcn
import numpy as np

description="test linear model"

def test_linear_operator():
    W0 = np.array([[1, -1, 5],
                   [0, 14, -9]])
    b0 = np.array([1, 4, -1])
    linear = fcn.Linear(W0, b0)
    y = linear(np.array([[1, -12]]))
    assert (y.value == np.array([[2, -165, 112]])).all()

def run():
    test_linear_operator()


if __name__ == "__main__":
    run()
