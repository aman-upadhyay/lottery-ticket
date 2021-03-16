import numpy as np

p = np.load("mnist_conv2D/mnist_conv2D_data/trial1/2/same_init/final/layer3.npy")
m = np.load("mnist_conv2D/mnist_conv2D_data/trial1/2/same_init/masks/layer3.npy")
print(m)
print(p)