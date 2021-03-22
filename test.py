import numpy as np
import tensorflow as tf


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


# no_of_weights_left(300)
kernel_initializer = tf.contrib.layers.xavier_initializer(uniform=False)
weights1 = tf.get_variable(
    name='w1',
    shape=[3, 3, 3, 4],
    initializer=kernel_initializer)

weights2 = tf.get_variable(
    name='w2',
    shape=[1, 1, 3, 0],
    initializer=kernel_initializer)

print(weights1, weights2)
tf.concat([weights1, weights2], axis=2)
