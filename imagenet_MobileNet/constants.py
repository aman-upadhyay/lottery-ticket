"""Constants for Conv2D CIFAR experiment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
from tqdm import tqdm
from foundations import paths
from cifar_MobileNet import locations
import tensorflow as tf

HYPERPARAMETERS = {}

CIFAR_LOCATION = locations.CIFAR_LOCATION

OPTIMIZER_FN = functools.partial(tf.train.GradientDescentOptimizer, .1)
# skip BN, avg pooling, skip layers
PRUNE_PERCENTS = {'layer0': .2, 'layer1': 0.2, 'layer2': 0.2, 'layer3': 0.2, 'layer4': 0.2, 'layer5': 0.2,
                  'layer6': 0.2, 'layer7': 0.2, 'layer8': 0.2, 'layer9': 0.2, 'layer10': 0.2, 'layer11': 0.2,
                  'layer12': 0.2, 'layer13': 0.2, 'layer14': 0.2, 'layer15': 0.2, 'layer16': 0.2, 'layer17': 0.2,
                  'layer18': 0.2, 'layer19': 0.2, 'layer20': 0.2, 'layer21': 0.2, 'layer22': 0.2, 'layer23': 0.2,
                  'layer24': 0.2, 'layer25': 0.2, 'layer26': 0.2}

prune_iteration = 8

TRAINING_LEN = ('epochs', 50)

EXPERIMENT_PATH = locations.EXPERIMENT_PATH

pbar = tqdm(total=TRAINING_LEN[1] * (prune_iteration + 1), ascii=".>=", desc="Pruning (1 iteration is 1 epoch)")


def graph(category, filename):
    return os.path.join(EXPERIMENT_PATH, 'graphs', category, filename)


def initialization(level):
    return os.path.join(EXPERIMENT_PATH, 'weights', str(level), 'initialization')


def trial(trial_name):
    return paths.trial(EXPERIMENT_PATH, trial_name)


def run(trial_name, level, experiment_name='same_init', run_id=''):
    return paths.run(trial(trial_name), level, experiment_name, run_id)
