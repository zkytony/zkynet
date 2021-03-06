# Copyright 2022 Kaiyu Zheng
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os, sys
ABS_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ABS_PATH, '../'))

from zkynet.framework import cg, op
from zkynet.visual import plot_cg
import jax.numpy as jnp
import time

description="testing forward construction for the computational graph framework"

class MyTestModel1(cg.Module):
    """A rather simple function that represents:

    f(x,w) = (x+w)*x^2

    where x is an input and w is a parameter.
    """
    def __init__(self, w0=1.):
        w0 = jnp.array([w0])
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

class CompositeModel_WeightSharing_DifferentInputs(cg.Module):
    """used to test composition"""
    def __init__(self):
        super().__init__(inputs=(cg.Variable("x1"),
                                 cg.Variable("x2")))
        # I expect the weights in the two may differ
        self._m1 = MyTestModel1(w0=2.)

    def call(self, x1, x2):
        a = self._m1(x1)
        b = self._m1(x2)
        return op.add(a, b)

class CompositeModel_NoWeightSharing_SameInputs(cg.Module):
    """used to test composition"""
    def __init__(self):
        super().__init__(inputs=(cg.Variable("x1"),))
        # I expect the weights in the two may differ
        self._m1 = MyTestModel1()
        self._m2 = MyTestModel1()

    def call(self, x1):
        a = self._m1(x1)
        b = self._m2(x1)
        return op.add(a, b)

class CompositeModel_WeightSharing_SameInputs(cg.Module):
    """used to test composition"""
    def __init__(self):
        super().__init__(inputs=(cg.Variable("x1"),))
        # I expect the weights in the two may differ
        self._m1 = MyTestModel1()

    def call(self, x1):
        a = self._m1(x1)
        b = self._m1(x1)
        return op.add(a, b)


def test_model1_forward():
    m = MyTestModel1()
    assert m.param("w").value == 1  # initial value

    x = jnp.array(3)
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
    x = jnp.array(3)
    result = m(x)
    plot_cg(result.root, save_path="/tmp/cg-simple", wait=2, title="test_visualize_cg (simple)")


def test_visualize_CompositeModel_NoWeightSharing_DifferentInputs():
    m = CompositeModel_NoWeightSharing_DifferentInputs()
    result = m(jnp.array(3), jnp.array(4))
    plot_cg(result.root, save_path="/tmp/cg-comp-NN", wait=2, title="test_visualize_**No**WeightSharing_**Different**Inputs")

def test_visualize_CompositeModel_WeightSharing_DifferentInputs():
    m = CompositeModel_WeightSharing_DifferentInputs()
    result = m(jnp.array(3), jnp.array(3))
    plot_cg(result.root, save_path="/tmp/cg-comp-YN", wait=2, title="test_visualize_CompositeModel_WeightSharing_**Different**Inputs")

def test_visualize_CompositeModel_NoWeightSharing_SameInputs():
    m = CompositeModel_NoWeightSharing_SameInputs()
    result = m(jnp.array(3), jnp.array(4))
    plot_cg(result.root, save_path="/tmp/cg-comp-NY", wait=2, title="test_visualize_CompositeModel_**No**WeightSharing_**Same**Inputs")

def test_visualize_CompositeModel_WeightSharing_SameInputs():
    m = CompositeModel_WeightSharing_SameInputs()
    result = m(jnp.array(3), jnp.array(4))
    plot_cg(result.root, save_path="/tmp/cg-comp-YY", wait=2, title="test_visualize_CompositeModel_WeightSharing_**Same**Inputs")


def run():
    test_model1_forward()
    test_visualize_cg()
    test_visualize_CompositeModel_NoWeightSharing_DifferentInputs()
    test_visualize_CompositeModel_WeightSharing_DifferentInputs()
    test_visualize_CompositeModel_NoWeightSharing_SameInputs()
    test_visualize_CompositeModel_WeightSharing_SameInputs()

if __name__ == "__main__":
    run()
