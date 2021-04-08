"""Run this script to prepare ImageNet and Tiny ImageNet datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from keras.datasets import cifar10 as CIFAR
from foundations import save_restore
from imagenet_MobileNet import locations
import numpy
from PIL import images
import six.moves.cPickle as pickle


def download(location=locations.imagenet_LOCATION):
    d = {}
    (d['x_train'], d['y_train']), (d['x_test'], d['y_test']) = CIFAR.load_data()
    save_restore.save_network(location, d)


def main():
    download()


if __name__ == '__main__':
    main()
