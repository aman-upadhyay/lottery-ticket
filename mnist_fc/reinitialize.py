"""The reinitialization experiment for Lenet 300-100 trained on MNIST."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from foundations import save_restore
from mnist_fc import constants
from mnist_fc import train as train_mnist
from foundations import paths
import numpy as np


def train(output_dir,
          mnist_location=constants.MNIST_LOCATION,
          training_len=constants.TRAINING_LEN,
          masks=None,
          initialization_distribution=None,
          same_sign=None):
    """Perform the reinitialization experiment.

  Using the masks from a previous run of the lottery ticket experiment, train
  a new, randomly reinitialized network.

  At most one of masks_location and masks_dictionary should be set. If both
  are None, then no masks are used.

  At most one of same_sign_location and same_sign_dictionary should be set. If
  both are None, then the same sign initialization strategy is not used.

  Args:
    output_dir: The directory to which the output should be written.
    mnist_location: The path to the NPZ file containing MNIST.
    training_len: How long to train the network.
    masks: The masks, if any, used to prune weights. Masks can come in
      one of three forms:
      * A dictionary of numpy arrays. Each dictionary key is the name of the
        corresponding tensor that is to be masked out. Each value is a numpy
        array containing the masks (1 for including a weight, 0 for excluding).
      * The string name of a directory containing one file for each
        mask (in the form of foundations.save_restore).
      * None, meaning the network should not be pruned.
    initialization_distribution: The distribution from which weights are sampled
      If the argument is None, the weights are samples from the default
      distribution.
      If the argument is a string, it is treated as the name of a directory
      whose file names are layer names and whose entries are one-dimensional numpy
      arrays of weights. The weights for each layer are randomly sampled from
      these arrays.
      If the argument is anything else, it is treated as a dictionary whose keys
      are layer names and whose values are numpy arrays as described above.
    same_sign: Whether to ensure each weight is initialized to the
      same sign as the weight in the original network. Only applies when
      initialization is not None. If this argument is not None, then it contains
      the previous network weights that are used to determine the signs to which
      the new network should be initialized. This argument can be provided as a
      dictionary or string path in the same fashion as masks.
  """
    masks = save_restore.standardize(masks)
    prev_weights = save_restore.standardize(same_sign)

    if initialization_distribution is None:
        presets = None
    else:
        initialization_distribution = save_restore.maybe_restore(
            initialization_distribution)

        # The preset weights should be randomly sampled from the values of
        # initialization. They should be the same shape as the masks.
        presets = {}
        for k, mask in masks.items():
            init = initialization_distribution[k]

            # Weights have the same sign as those in the original networks.
            if prev_weights:
                positive = np.random.choice(init[init > 0], mask.shape)
                negative = np.random.choice(init[init < 0], mask.shape)
                presets[k] = np.where(prev_weights > 0, positive, negative)

            # Weights are randomly sampled.
            else:
                presets[k] = np.random.choice(init, mask.shape)

    train_mnist.train(
        output_dir=output_dir,
        mnist_location=mnist_location,
        training_len=training_len,
        presets=presets,
        masks=masks)


for trial in range(1, 21):
    for level in range(0, 31):
        for run in range(1, 11):
            masks = paths.masks(constants.run(trial, level))
            output = constants.run(trial, level, 'name of the exp', run)

            train(output_dir=output, masks=masks, initialization_distribution=constants.initialization(level), same_sign=True)
