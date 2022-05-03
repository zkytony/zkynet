# zkynet
exploring deep learning basics

## Setup

1. Run setup.sh to create and activate a designated virtualenv,
   if you so desire.

    You should install:
    ```
    pip install torch torchvision
    ```


2. Install [JAX](https://github.com/google/jax)
   Follow the steps for [GPU (CUDA)](https://github.com/google/jax#pip-installation-gpu-cuda).
   Essentially:
   1. Make sure your CUDA and cuDNN versions are correct
   2. Then run a pip install command to install `jax[cuda]`. As of 05/02/2022:
      ```
      pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html  # Note: wheels only available on linux.
      ```

3. (Optional): Run `download_datasets.sh`

    This will download several kaggle datasets.
