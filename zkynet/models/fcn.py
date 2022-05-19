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
Linear layer
"""
from zkynet.framework import cg, op

class Linear(cg.Module):
    """
    A linear layer maps a vector of dimension (n,1)
    to a vector of dimension (p,1) by:

    :math:`f(x) = x^TW + b`

    where:

        - :math:`x \in \mathbb{R}^{1\times n}`
        - :math:`W \in \mathbb{R}^{n\times p}`
        - :math:`b \in \mathbb{R}^{1\times p}`

    Here, we accept x in the form of a matrix X of shape (D,n),
    so the linear mapping can be efficiently computed for
    many data points.
    """
    def __init__(self, init_weights, init_bias):
        """
        Args:
            init_weights (np.array): matrix of shape (n,p)
            init_bias (np.array): vector of shape (p,)
        """
        # Verify shape
        n, p = init_weights.shape
        if p != init_bias.shape[0]:
            raise ValueError(f"Invalid shape; Weight matrix"
                             "is {(n,p)} while bias is {init_bias.shape}")

        super().__init__(inputs=(cg.Variable("X"),),
                         params=(cg.Parameter("W", init_weights),
                                 cg.Parameter("b", init_bias)))

    @property
    def input_dim(self):
        """returns :math:`n`, the input dimension"""
        return self.param("W").value.shape[0]

    @property
    def output_dim(self):
        """returns :math:`p`, the output dimension"""
        return self.param("W").value.shape[1]

    def call(self, x):
        y = op.add(op.dot(x, self.param("W")), self.param("b"))
        return y
