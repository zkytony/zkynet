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

import uuid
import jax.numpy as jnp

def unique_id(length=6):
    return uuid.uuid4().hex[:length].upper()


def fullname(o):
    klass = o.__class__
    module = klass.__module__
    if module == 'builtins':
        return klass.__qualname__ # avoid outputs like 'builtins.str'
    return module + '.' + klass.__qualname__


def backpropdot(t1, t2):
    """returns the product of two tensors t1 and t2
    that can deal with the case when t1 or t2 are
    vectors or matrices instead. Used in backprop."""
    if len(t1.shape) <= 2 and len(t2.shape) <= 2:
        return jnp.dot(t1, t2)
    elif len(t1.shape) <= 4 and len(t2.shape) <= 4:
        return jnp.tensordot(t1, t2)
    else:
        # this seems to be the right call for tensor product
        # (though I am not sure why)
        return jnp.multiply(t1, t2)
