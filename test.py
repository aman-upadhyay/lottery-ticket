import tensorflow as tf
from foundations.model_base import ModelBase
from datasets import dataset_mnist
from mnist_fc import constants
import numpy as np

dset = dataset_mnist.DatasetMnist(
    "mnist_fc/data/mnist",
    flatten=False,)
# input_mnist = input_mnist[1, :, :]
print(dset)
test = ModelBase()
out = test.conv2D(name="temp", inputs=input_mnist, filters=3, kernel_size=(3, 3), strides=1,
                  kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
