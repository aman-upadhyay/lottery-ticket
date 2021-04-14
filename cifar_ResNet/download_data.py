"""Run this script to download CIFAR and FashionCIFAR datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from keras.datasets import cifar100 as CIFAR
from foundations import save_restore
from cifar_ResNet import locations


def download(location=locations.CIFAR_LOCATION):
    d = {}
    (d['x_train'], d['y_train']), (d['x_test'], d['y_test']) = CIFAR.load_data()
    save_restore.save_network(location, d)


def main():
    download()


if __name__ == '__main__':
    main()
