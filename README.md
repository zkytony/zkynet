# zkynet: exploring deep learning basics through implementations.

## Examples

### Proof-of-Concept Model

1. Define a simple model for the function `f(x) = (x+w)*x^2` where `x` is an input and `w` is a parameter

    ```python
    from zkynet.framework import cg, op

    class SimpleModel(cg.Module):
        """A rather simple function that represents:

        f(x,w) = (x+w)*x^2

        where x is an input and w is a parameter.
        """
        def __init__(self, w0=1):
            super().__init__(inputs=(cg.Variable("x"),),
                             params=(cg.Parameter("w", w0),))

        def call(self, x):
            a = op.add(x, self.param("w"))
            b = op.square(x)
            c = op.mult(a, b)
            return c
    ```
    Notice how the `call()` function defines the forward pass of
    the model, where operations come from `zkynet.framework.op`
    (short for `zkynet.framework.operations`). These are operations
    specifically designed to work with our computational graph framework.

    **Relation to PyTorch:** (1) the `call()` function
    is like `forward()` in a PyTorch nn.Module. (2) PyTorch
    uses Tensors as its representation of values, which
    have many built-in operations. We don't rely on Tensor
    (we are building from basic scratch) so we use our own
    operators.


2. Forward pass:
   ```python
   m = SimpleModel()
   result = m(3)
   result.value
   # 36
   ```
   Here, `m(3)` calls the model and performs a forward pass,
   with the input `x` set to value `3`. The output `result`
   is of type **ModuleGraph** which represents a computational
   graph that is grounded to the given input.

   Note that each call produces an independent computational
   graph. Namely:
   ```python
   result1 = m(3)
   result2 = m(3)
   assert result1 != result2
   assert result1.root != result2.root
   ```
   Each call of the model generates a _call_id_, which is used
   to distinguish between the computational graphs generated
   for each call.

   If you think of defining the model class as writing a
   "template" of how inputs are associated to produce an
   output, then a computational graph is an instantiation
   of that template with all the placeholder inputs are filled
   with concrete, given values.

   Note that you could pass in a vector too:
   ```python
   import numpy as np

   result = m(np.array([3, 4, 5]))
   result.value
   # array([ 36,  80, 150])
   ```

3. Backprop & gradients:

    ```python
    result = m(3)
    result.back()  # backprop; accumulate gradients
    result.grad(m.param("w"))  # obtain dm/dw
    # 9
    result.grad(m.input("x"))  # obtain dm/dx
    # x
    ```


## Installation

Run `setup.sh` to create and activate a designated virtualenv.
The first time the virtualenv is activated, the script will install
dependency packages.


### Install JAX
For some parts of the codebase, you may need to use JAX.

Install [JAX](https://github.com/google/jax).
   Follow the steps for [GPU (CUDA)](https://github.com/google/jax#pip-installation-gpu-cuda).
   Essentially:

   1. Make sure your CUDA and cuDNN versions are correct
   2. Then run a pip install command to install `jax[cuda]`. As of 05/02/2022:
      ```
      pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html  # Note: wheels only available on linux.
      ```


### (Optional): Download Kaggle Datasets

Run `download_datasets.sh`

This will download several kaggle datasets.



## Progress tracker

 - [X] Implement basic feed-forward network
 - [ ] Be able to train the network using gradient descent
   - [-] Implement a framework for automatic differentiation
        - [X] computational graph forward construction (05/06/2022)
        - [X] backward gradient accumulation (05/09/2022)
        - [ ] log space
        - [ ] vectorization tests

 - [ ] Implement convolutional neural network
 - [ ] Implement recurrent neural network
 - [ ] Implement auto-encoder
 - [ ] Implement transformer





## APPENDIX: JAX
The JAX library provides a high-level interface
`jax.numpy` and a low-level interface `jax.lax`. The
low-level interface is stricter, but often more
powerful. Functions in `jax.numpy` eventually get
passed down to calls to `jax.lax` functions.

In JAX, arrays are ALWAYS immutable.

The main feature of JAX is "Just-In-Time" compilation,
which means code gets compiled the first time they
are run. Doc says:
>Not all JAX code can be JIT compiled, as it requires array shapes to be static & known at compile time

The way JAX can do JIT is because it expresses
its operations in terms of XLA (the Accelerated Linear
Algebra compiler).

In fact, people use JAX as a deep learning framework
too - alongside PyTorch. See [[this reddit post](https://www.reddit.com/r/MachineLearning/comments/shsfkm/d_current_state_of_jax_vs_pytorch/hv4h3k7/).

### autograd
[autograd](https://github.com/HIPS/autograd) is a library that differentiates native Python and Numpy code.
Pytorch either uses this library or implements something like it [torch.autograd](https://pytorch.org/docs/stable/autograd.html)
that works on Pytorch's tensors (instead of native python / numpy data structures).

Autograd's `grad` function takes in a function, and gives you a function that
computes its derivative.

### Thoughts
一开始我觉得Autograd很神奇，跟tensorflow不一样。但是，实际上他们是一样的。
背后都有一个operation framework，只不过tensorflow是用自定义的framework，
而autograd利用的numpy和python自带的，比较轻便罢了。本质上的原理没有区别。
