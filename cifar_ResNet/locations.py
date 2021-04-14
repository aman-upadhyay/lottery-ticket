"""Root locations for ResNet CIFAR experiments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# CIFAR is stored as a directory containing four npy files:
#   x_train.npy, x_test.npy, y_train.npy, y_test.npy
# See datasets/dataset_CIFAR.py for details.

# Originally from https://s3.amazonaws.com/img-datasets/CIFAR.npz
CIFAR_LOCATION = 'data/CIFAR'

EXPERIMENT_PATH = 'CIFAR_ResNet_data'
