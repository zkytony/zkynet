import os, sys
ABS_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ABS_PATH, '../'))

from zkynet.framework import cg
from zkynet.framework import op
import numpy as np

def test_ops():
    pass

if __name__ == "__main__":
    test_ops()
