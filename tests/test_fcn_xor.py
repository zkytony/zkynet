# Use FCN to represent the XOR function.
# The network comes from figure 6.2 of
# the Deep Learning textbook.

import os, sys
ABS_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ABS_PATH, '../'))
from zkynet.models.fcn import FCN, FCNLayer, LinearLayer, ReLU, Identity
import numpy as np

def test_random():
    """the weights are random"""
    np.random.seed(4)
    network = FCN([FCNLayer.build((2,2), ReLU),
                   FCNLayer.build((2,1), Identity)])
    print("----------")
    print(network)
    print("----------")
    print(network(np.array([[1,0]])))


def test_correct():
    """the weights here are set according to the book"""
    np.random.seed(4)
    network = FCN([FCNLayer.build((2,2), ReLU,
                                  weights=np.array([[1, 1], [1, 1]]),
                                  bias=np.array([0, -1])),
                   FCNLayer.build((2,1), Identity,
                                  weights=np.array([[1], [-2]]),
                                  bias=np.array([0]))])
    print("----------")
    print(network)
    print("----------")
    print(network(np.array([[1,0]])))

if __name__ == "__main__":
    test_random()
    test_correct()
