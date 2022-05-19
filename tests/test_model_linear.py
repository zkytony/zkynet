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

"""
Test linear model
"""
import os, sys
ABS_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ABS_PATH, '../'))

from zkynet.models import fcn
import jax.numpy as jnp
import torch

description="test linear model"

def test_linear_operator():
    W0 = jnp.array([[1, -1, 5],
                    [0, 14, -9]], dtype=jnp.float32)
    b0 = jnp.array([1, 4, -1], dtype=jnp.float32)
    linfn = fcn.Linear(W0, b0)
    x = jnp.array([1, -12], dtype=jnp.float32)
    y = linfn(x)
    assert (y.value == jnp.array([[2, -165, 112]])).all()

    y.back()
    print(y.grad(linfn.param("W")))


def run():
    test_linear_operator()


if __name__ == "__main__":
    run()
