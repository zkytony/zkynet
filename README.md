# zkynet: Exploring deep learning basics through implementations

An autodiff framework with a PyTorch-like API that supports tensor inputs and outputs.

Key points to note:
- All inputs to functions represented using our computational graph are
  expected to be [JAX arrays](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.array.html).
  
- This is designed completely from scratch after reading and understanding the autodiff algorithm. It is in no way optimal but it is an incredibly useful exercise. 


## Concepts

The overall design principles of this framework is that:

- A **function** is an _abstract template_ that maps ordered **inputs** (`Variable`)
  to an output subject to some internal **parameters** (`Parameter`));

  In order to define the function, the values of these parameters
  would need to be tracked of inside the Function object. For example,
  think of a neural network model. The model carries parameters. But
  before we apply it to some real inputs, the model exists as a floating
  blackbox; it isn't _grounded_ to any input values.

- There are two kinds of functions, operators (`Operator`) and
    modules (`module`).

    An **operator** is a function that we intend to hard code its
    derivatives.  Such functions output numbers or arrays.

    A **module** is a function that is intended to be
    user-defined, (maybe) complicated functions.  There is a _flat_
    _grounded_ computational graph corresponding to a module,
    created upon the module is called. This computational graph,
    as explained next, consists of nodes that represent inputs
    or operators (but not modules - that's why we say it's flat).


- When concrete input values are provided for a function call, a
  **computational graph** (a DAG) is created. We represent the
  graph using nodes (`Node`). A **node** can always be
  regarded as an instantiation of a particular Input to a
  function. It carries a **value**.  Since it is a DAG, a node can
  have multiple children and multiple parents.

- We distinguish two node types: input node (`InputNode`) and
    operator node (`OperatorNode`).

    The **input node** is a leaf node. It literally represents a
    leaf node on the DAG.

    The **operator node** is not a leaf node.  Both
    should be grounded with values.  The value of the operator
    node represents the output of the function (specifically, an
    Operator) under some input node instantiation.

-  A **ModuleGraph** is a computational graph that
    is grounded when a Module is called. It stores
    a flat computational graph (by 'flat' we mean
    that its internal OperatorNodes should only be
    Operators.)

    Note that since a Module's call may involve
    calling another module, we don't actually
    create a graph for that module. We only care
    about the trigger function (i.e. the first Module),
    similar to CallSessionManager.


-   In order to enforce independence between computational
    graphs from different calls, **CallSessionManager** will
    maintain the **call ID** of the current call, which is assigned
    to all nodes that are created during the call.

    It will clear the call ID if the call to the trigger
    function is finished (the trigger function is the first
    function that is called, which is likely a user-defined
    model).

    Additionally, it stores InputNodes that have been
    created (identified by its ID), so that subsequent
    calls to the 'to_node' method of Input do not create
    new ones (which may have wrong parent/children relationships)
    but reuse the ones stored here.

- Each Function or Input has a name and a functional name.


  **Functional name:** The name that identifies the ROLE this template
  object plays in the definition of a function; For example,
  if self is an Input, then this is the name that identifies
  both the function and the role this input plays to that function.


  **Name:** The string that identifies the VARAIBLE name (or ENTITY)
  that this template object represents. For example,
  we could have two Function objects that represent
  the same function but we care about their outputs
  as separate variables. Then, these two Function objects
  should have the same 'functional_name' but different
  'name'.



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

3. Backprop (autodiff) & gradients:

    ```python
    result = m(3)
    result.back()  # backprop; accumulate gradients
    result.grad(m.param("w"))  # obtain dm/dw
    # 9
    result.grad(m.input("x"))  # obtain dm/dx
    # 33
    ```

    Note that backpropagation works with vector, matrix, or tensor inputs.
    For example:
    ```python
    # vector input
    x = jnp.array([3., 4., 5.])
    result = m(x)
    result.back()
    result.grad(m.input("x"))
    # Out[7]:
    # DeviceArray([[33.,  0.,  0.],
    #              [ 0., 56.,  0.],
    #              [ 0.,  0., 85.]], dtype=float32)

    result.grad(m.param("w"))
    # DeviceArray([[ 9.],
    #              [16.],
    #              [25.]], dtype=float32)
    ```
    See [`test_cg_backward.py`](./tests/test_cg_backward.py) for examples
    of matrix and tensor inputs.



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
The interface is simple when implementing an operator (of class `Operator`). You only
need to implement the forward-pass in `_op_impl` (JAX takes care of the Jacobian
using `jacrev`; we only use `jacrev` at the operator-level, and
implement _our own_ automatic differentiation algorithm for module-level differentiation.)
Below are examples for `Add`, `Multiply`, `Square`, and `Dot`.

```python
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
```
Note that every time when using an operator to compose a function,
you need to create a new object for it. For convenience, you
could define utility functions like:
```python
def add(a, b):
    return Add()(a, b)

def mult(a, b):
    return Multiply()(a, b)

def square(x):
    return Square()(x)

def dot(a, b):
    return Dot()(a, b)
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
      
### Install Flax (not used for now)
Check out [the flax installation docs](https://flax.readthedocs.io/en/latest/installation.html):
```
pip install flax
```


### (Optional): Download Kaggle Datasets

Run `download_datasets.sh`

This will download several kaggle datasets.



## Progress tracker

   1. Build a framework for automatic differentiation
      - Design with a PyTorch-like API
      - Able to do forward pass
      - Able to do backward pass (backprop) for scalar input/output
      - Extension to vector, matrix, tensor input/output (using JAX)


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
