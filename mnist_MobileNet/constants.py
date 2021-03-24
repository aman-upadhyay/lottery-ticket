"""Constants for Conv2D MNIST experiment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
from tqdm import tqdm
from foundations import paths
from mnist_MobileNet import locations
import tensorflow as tf

HYPERPARAMETERS = {}

MNIST_LOCATION = locations.MNIST_LOCATION

OPTIMIZER_FN = functools.partial(tf.train.GradientDescentOptimizer, .1)

PRUNE_PERCENTS = {'layer0': .2}

prune_iteration = 3

TRAINING_LEN = ('epochs', 50)

EXPERIMENT_PATH = locations.EXPERIMENT_PATH

pbar = tqdm(total=TRAINING_LEN[1]*prune_iteration, ascii="=>", desc="Pruning (1 iteration is 1 epoch)")

def graph(category, filename):
    return os.path.join(EXPERIMENT_PATH, 'graphs', category, filename)


def initialization(level):
    return os.path.join(EXPERIMENT_PATH, 'weights', str(level), 'initialization')


def trial(trial_name):
    return paths.trial(EXPERIMENT_PATH, trial_name)


def run(trial_name, level, experiment_name='same_init', run_id=''):
    return paths.run(trial(trial_name), level, experiment_name, run_id)
