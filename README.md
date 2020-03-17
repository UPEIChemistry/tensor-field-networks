## How to Install

requirements: `atomic-images`, `tensorflow 2.0`

This installation guide expects the user to understand pip and have it installed. This package depends on 
`atomic-images` available [here](https://github.com/UPEIChemistry/atomic-images).
Clone `atomic-images` using `git clone git@github.com:UPEIChemistry/atomic-images.git`, then checkout the `tf_2.0` branch
by using
`git checkout tf_2.0`. Once this branch is checked out, use `pip install -e ./atomic-images` to install the package.
To install tensor-field-networks, start by cloning the repo with 
`git clone git@github.com:UPEIChemistry/tensor-field-networks.git`, 
followed by using pip: `pip install -e ./tensor-field-networks`. The setup.py script contained in this package 
should install tensorflow 2, numpy, and any other 'official' dependencies. Be sure to install `tensorflow-gpu==2.0.0` 
and CUDA/cudNN if you intend to use this code on a GPU (which is recommended for the performance boost).

# Tensor Field Networks

Tensor Field Networks (TFN) are **Rotationally Equivariant Continuous Graph Convolution Neural Networks** which are 
 capable of inputing continuous 3D point-clouds (e.g. molecules) and making scalar, vector, and higher order tensor 
 predictions which rotate with the original input point-cloud ([Thomas et. al., 2018](https://arxiv.org/abs/1802.08219)).

Ignoring the **continuous convolution** part, this means that TFNs are capable of knowing when an image has been 
rotated, something vanilla convolution nets are not capable of. For example, a traditional conv. net trained to 
recognize cats on **non-rotated images** would not identify a cat in the second picture:

![cat](tutorials/cat_pic.png) ![cat_rotated](tutorials/cat_pic_rotated.png)

While TFNs will still identify a cat in the rotated image, trained only on images in a single orientation. To see a
demonstration of this equivariance, and a further explanation of TFNs, checkout the Jupyter notebook located in the
`tutorials` directory. If the user is not familiar with using Jupyter notebooks, they can read up on them 
[here](https://jupyter.readthedocs.io/en/latest/content-quickstart.html).
