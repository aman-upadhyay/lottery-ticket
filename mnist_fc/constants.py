"""Constants for fully-connected MNIST experiments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
from foundations import paths
from mnist_fc import locations
import tensorflow as tf

HYPERPARAMETERS = {'layers': [(300, tf.nn.relu), (100, tf.nn.relu), (10, None)]}

MNIST_LOCATION = locations.MNIST_LOCATION

# FASHIONMNIST_LOCATION = locations.FASHIONMNIST_LOCATION

OPTIMIZER_FN = functools.partial(tf.train.GradientDescentOptimizer, .1)

PRUNE_PERCENTS = {'layer0': .2, 'layer1': .2, 'layer2': .1}

TRAINING_LEN = ('iterations', 50000)

EXPERIMENT_PATH = locations.EXPERIMENT_PATH


def graph(category, filename):
    return os.path.join(EXPERIMENT_PATH, 'graphs', category, filename)


def initialization(level):
    return os.path.join(EXPERIMENT_PATH, 'weights', str(level), 'initialization')


def trial(trial_name):
    return paths.trial(EXPERIMENT_PATH, trial_name)


def run(trial_name, level, experiment_name='same_init', run_id=''):
    return paths.run(trial(trial_name), level, experiment_name, run_id)
