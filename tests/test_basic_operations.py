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
Tests for operations under zkynet/framework/operations.py
"""
import os, sys
ABS_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ABS_PATH, '../'))

from zkynet.framework import cg, op
import jax.numpy as jnp

description="testing basic operations"

def test_add_operator_gradient():
    add_op = op.Add()
    # a + b; dfda = 1
    dfda_fn = add_op.gradfn(cg.Variable("a"))
    assert dfda_fn(jnp.array(1.), jnp.array(2.)).value == 1

def test_multiply_operator_gradient():
    mult_op = op.Multiply()
    # a + b; dfda = 1
    dfda_fn = mult_op.gradfn(cg.Variable("a"))
    assert dfda_fn(jnp.array(1.), jnp.array(2.)).value == 2
    dfdb_fn = mult_op.gradfn(cg.Variable("b"))
    assert dfdb_fn(jnp.array(1.), jnp.array(2.)).value == 1

def test_square_operator_gradient():
    square_op = op.Square()
    # a + b; dfda = 1
    dfdx_fn = square_op.gradfn(cg.Variable("x"))
    assert dfdx_fn(jnp.array(2.)).value == 4

def test_node_grad_function():
    mult_op = op.Multiply()
    result = mult_op(jnp.array(3.), jnp.array(4.))
    num = result.grad(result.children[0])
    assert num == 4

def run():
    test_add_operator_gradient()
    test_multiply_operator_gradient()
    test_square_operator_gradient()
    test_node_grad_function()


if __name__ == "__main__":
    run()
