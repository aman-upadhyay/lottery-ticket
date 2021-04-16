"""The lottery ticket experiment for ResNet trained on CIFAR100."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
from datasets import dataset_cifar
from foundations import experiment
from foundations import ResNet34
from foundations import paths
from foundations import pruning
from foundations import save_restore
from foundations import trainer
from cifar_ResNet import constants


def train(output_dir,
          CIFAR_location=constants.CIFAR_LOCATION,
          training_len=constants.TRAINING_LEN,
          iterations=constants.prune_iteration,
          experiment_name='same_init',
          presets=None,
          permute_labels=False,
          train_order_seed=None):
    """Perform the lottery ticket experiment.

  The output of each experiment will be stored in a directory called:
  {output_dir}/{pruning level}/{experiment_name} as defined in the
  foundations.paths module.

  Args:
    output_dir: Parent directory for all output files.
    CIFAR_location: The path to the NPZ file containing CIFAR.
    training_len: How long to train on each iteration.
    iterations: How many iterative pruning steps to perform.
    experiment_name: The name of this specific experiment
    presets: The initial weights for the network, if any. Presets can come in
      one of three forms:
      * A dictionary of numpy arrays. Each dictionary key is the name of the
        corresponding tensor that is to be initialized. Each value is a numpy
        array containing the initializations.
      * The string name of a directory containing one file for each
        set of weights that is to be initialized (in the form of
        foundations.save_restore).
      * None, meaning the network should be randomly initialized.
    permute_labels: Whether to permute the labels on the dataset.
    train_order_seed: The random seed, if any, to be used to determine the
      order in which training examples are shuffled before being presented
      to the network.
  """

    make_model = functools.partial(ResNet34.ResNet34)

    # Define model and dataset functions.
    def make_dataset():
        return dataset_cifar.DatasetCIFAR(
            CIFAR_location,
            inc_dim=False,
            flatten=False,
            permute_labels=permute_labels,
            train_order_seed=train_order_seed)

    # Define a training function.
    def train_model(sess, level, dataset, model):
        params = {
            'test_interval': 100,
            'save_summaries': True,
            'save_network': True,
        }

        return trainer.train(
            sess,
            dataset,
            model,
            constants.OPTIMIZER_FN,
            training_len,
            prog=constants.pbar,
            output_dir=paths.run(output_dir, level, experiment_name),
            **params)

    # Define a pruning function.
    prune_masks = functools.partial(pruning.prune_by_percent,
                                    constants.PRUNE_PERCENTS)

    # Run the experiment.
    experiment.experiment(
        make_dataset,
        make_model,
        train_model,
        prune_masks,
        iterations,
        presets=save_restore.standardize(presets))


x = 0
while True:
    path = "CIFAR_ResNet_data/trial{}".format(str(x))
    isFile = os.path.isdir(path)
    if isFile:
        x = x + 1
    else:
        train(constants.trial(x))
        constants.pbar.close()
        exit()
