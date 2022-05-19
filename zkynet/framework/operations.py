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
Hard-coded operations

The operations here are the most basic building
blocks. For more complex operations, look into
specific models in zkynet.models
"""
import jax.numpy as jnp
from jax import jacrev
from .computation_graph import Operator, Variable, Module

class Identity(Operator):
    def __init__(self):
        super().__init__(inputs=(Variable("x"),))

    def _op_impl(self, x):
        return jnp.identity(x)


class Add(Operator):
    def __init__(self):
        super().__init__(inputs=(Variable("a"), Variable("b")))

    def _op_impl(self, a, b):
        return a + b


class Multiply(Operator):
    def __init__(self):
        super().__init__(inputs=(Variable("a"), Variable("b")))

    def _op_impl(self, a, b):
        return a * b  # element wise multiplication


class Square(Operator):
    def __init__(self):
        super().__init__(inputs=(Variable("x"),))

    def _op_impl(self, x):
        return jnp.square(x)


class Dot(Operator):
    def __init__(self):
        super().__init__(inputs=(Variable("a"), Variable("b")))

    def _op_impl(self, a, b):
        return jnp.dot(a, b)


def add(a, b):
    return Add()(a, b)

def mult(a, b):
    return Multiply()(a, b)

def square(x):
    return Square()(x)

def identity(x):
    return Identity()(x)

def dot(a, b):
    return Dot()(a, b)
