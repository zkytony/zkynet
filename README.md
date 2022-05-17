# zkynet: Exploring deep learning basics through implementations

An autodiff framework with a PyTorch-like API that supports tensor inputs and outputs.

**Caveat: the tensor input may throw exceptions**

Key points to note:
- all inputs to functions represented using our computational graph are
  expected to be [JAX arrays](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.array.html).


## Examples

### Proof-of-concept model

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
    # 33
    ```

### Composite models and visualization
You can visualize a computational graph as
follows:
```python
from zkynet.visual import plot_cg

m = SimpleModel()
result = m(3)
plot_cg(result.root, wait=2, title="simpel model")
```
This shows:

![cg-simple](https://user-images.githubusercontent.com/7720184/167731761-bf651910-1a2a-463e-9384-41a4295c9f10.png)


As a more useful example, we will visualize the computational graph
for a few models. First, let's define a few more complex
models that are composed of the `SimpleModel`.
There are four cases in total:

**NO weight sharing, NO input sharing:**
```python
class CompositeModel_NoWeightSharing_DifferentInputs(cg.Module):
    """used to test composition"""
    def __init__(self):
        super().__init__(inputs=(cg.Variable("x1"),
                                 cg.Variable("x2")))
        # I expect the weights in the two may differ
        self._m1 = SimpleModel()
        self._m2 = SimpleModel()

    def call(self, x1, x2):
        a = self._m1(x1)
        b = self._m2(x2)
        return op.add(a, b)

m = CompositeModel_NoWeightSharing_DifferentInputs()
result = m(3, 4)
plot_cg(result.root, wait=2, title="NoWeightSharing_DifferentInputs")
```
This generates:

![cg-comp-NN](https://user-images.githubusercontent.com/7720184/167731793-d814fa88-3a23-44ae-92b5-6c7178718b47.png)


**YES Weight sharing, NO input sharing:**
```python
class CompositeModel_WeightSharing_DifferentInputs(cg.Module):
    """used to test composition"""
    def __init__(self):
        super().__init__(inputs=(cg.Variable("x1"),
                                 cg.Variable("x2")))
        # I expect the weights in the two may differ
        self._m1 = SimpleModel(w0=2)

    def call(self, x1, x2):
        a = self._m1(x1)
        b = self._m1(x2)
        return op.add(a, b)

m = CompositeModel_WeightSharing_DifferentInputs()
result = m(3, 3)
plot_cg(result.root, wait=2, title="test_visualize_CompositeModel_WeightSharing_**Different**Inputs")
```
This generates:

![cg-comp-YN](https://user-images.githubusercontent.com/7720184/167731827-8fe3555c-94f8-461c-92e1-48b4928eb64b.png)


**NO weight sharing, YES sharing inputs:**
```python
class CompositeModel_NoWeightSharing_SameInputs(cg.Module):
    """used to test composition"""
    def __init__(self):
        super().__init__(inputs=(cg.Variable("x1"),))
        # I expect the weights in the two may differ
        self._m1 = SimpleModel()
        self._m2 = SimpleModel()

    def call(self, x1):
        a = self._m1(x1)
        b = self._m2(x1)
        return op.add(a, b)

m = CompositeModel_NoWeightSharing_SameInputs()
result = m(3, 4)
plot_cg(result.root, wait=2, title="test_visualize_CompositeModel_**No**WeightSharing_**Same**Inputs")
```
This generates:

![cg-comp-NY](https://user-images.githubusercontent.com/7720184/167731855-7520836f-89ca-4eb6-8104-50c109c51b9b.png)




**YES weight sharing, YES sharing inputs:**
```python
class CompositeModel_WeightSharing_SameInputs(cg.Module):
    """used to test composition"""
    def __init__(self):
        super().__init__(inputs=(cg.Variable("x1"),))
        # I expect the weights in the two may differ
        self._m1 = SimpleModel()

    def call(self, x1):
        a = self._m1(x1)
        b = self._m1(x1)
        return op.add(a, b)

m = CompositeModel_WeightSharing_SameInputs()
result = m(3, 4)
plot_cg(result.root, wait=2, title="test_visualize_CompositeModel_WeightSharing_**Same**Inputs")
```
This generates:

![cg-comp-YY](https://user-images.githubusercontent.com/7720184/167731875-3d7f4476-e8a7-4037-b4a7-168dc336e77e.png)



### Writing an Operator
Below shows an example of implementing a multiplication
operator, called `Multiply`.

```python
from zkynet.framework import cg

class Multiply(cg.Operator):
    def __init__(self):
        super().__init__(inputs=(cg.Variable("a"), cg.Variable("b")))

    def call(self, a, b):
        return a.value * b.value

    def _gradfn(self, inpt):
        def _a_grad(a, b):
            return b.value
        def _b_grad(a, b):
            return a.value
        if inpt.short_name == "a":
            return _a_grad
        elif inpt.short_name == "b":
            return _b_grad
        else:
            raise ValueError(
                f"Unknown input for {self.functional_name}: {inpt.short_name}")
```

Note that the two methods to override are `call` (forward) and
`_gradfn` (backward).  The function `call` can return either a number
or array or an OperatorNode. The function `_gradfn` should return _a
function_ that represents the gradient function, which takes in the
same inputs as the operator.


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

## Troubleshooting JAX
If you experience the following error messages when performing a seemingly simple
operation such as `jnp.dot`,
```
2022-05-14 00:05:33.943054: E external/org_tensorflow/tensorflow/stream_executor/cuda/cuda_blas.cc:232] failed to create cublas handle: CUBLAS_STATUS_NOT_INITIALIZED
2022-05-14 00:05:33.943074: E external/org_tensorflow/tensorflow/stream_executor/cuda/cuda_blas.cc:234] Failure to initialize cublas may be due to OOM (cublas needs some free memory when you
 initialize it, and your deep-learning framework may have preallocated more than its fair share), or may be because this binary was not built with support for the GPU in your machine.
2022-05-14 00:05:33.943104: F external/org_tensorflow/tensorflow/compiler/xla/service/gpu/gemm_algorithm_picker.cc:324] Check failed: stream->parent()->GetBlasGemmAlgorithms(&algorithms)
Aborted (core dumped)
```
Then, according to [this github thread](https://github.com/google/jax/issues/7118),
there maybe something funky with your CUDA / JAX version stuff. And you should run
```
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```
although I am not sure if with that GPU still gets used? **YES. I CAN CONFIRM THAT IT IS USED; The memory is dynamically allocated.**
