# zkynet
exploring deep learning basics

## Setup

1. Run setup.sh to create and activate a designated virtualenv,
   if you so desire.



2. Install [JAX](https://github.com/google/jax)
   Follow the steps for [GPU (CUDA)](https://github.com/google/jax#pip-installation-gpu-cuda).
   Essentially:
   1. Make sure your CUDA and cuDNN versions are correct
   2. Then run a pip install command to install `jax[cuda]`.


3. (Optional): Run `download_datasets.sh`

    This will download several kaggle datasets.
