import time

import numpy as np
import sys
#import tensorflow as tf
from tqdm import tqdm


def weight():
    w = np.load("mnist_conv2D/mnist_conv2D_data/trial3/10/same_init/final/layer3.npy")
    print(w)
    print(w.shape)


def no_of_weights_left(x):
    y = 100
    print(0, x)
    for h in range(1, 30):
        y = y - 0.2 * y
        print(h, (y * x) / 100)

np.set_printoptions(threshold=sys.maxsize)
# no_of_weights_left(300)

x = np.load("cifar_ResNet/CIFAR_ResNet_data/trial1/1/same_init/final/layer1.npy")

print(x)