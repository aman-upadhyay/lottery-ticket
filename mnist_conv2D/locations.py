"""Root locations for Conv2D MNIST experiments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# MNIST is stored as a directory containing four npy files:
#   x_train.npy, x_test.npy, y_train.npy, y_test.npy
# See datasets/dataset_mnist.py for details.

# Originally from https://s3.amazonaws.com/img-datasets/mnist.npz
MNIST_LOCATION = 'data/mnist'

EXPERIMENT_PATH = 'mnist_fc_data'
