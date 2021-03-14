"""Run this script to download MNIST and FashionMNIST datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from keras.datasets import mnist
from foundations import save_restore
from mnist_conv2D import locations


def download(location=locations.MNIST_LOCATION):
    d = {}
    (d['x_train'], d['y_train']), (d['x_test'], d['y_test']) = mnist.load_data()
    save_restore.save_network(location, d)


def main():
    download()


if __name__ == '__main__':
    main()
