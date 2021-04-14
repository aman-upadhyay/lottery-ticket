"""Train the Lenet 300-100 model on MNIST.

Optionally initialize from saved initializations and masks.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datasets import dataset_mnist
from foundations import model_conv2D
from foundations import save_restore
from foundations import trainer
from foundations import union
from mnist_conv2D import constants
import tensorflow as tf


def train(output_dir,
          mnist_location=constants.MNIST_LOCATION,
          training_len=constants.TRAINING_LEN,
          masks=None,
          presets=None,
          train_order_seed=None):
    """Train the MNIST model, possibly with presets and masks.

  Args:
    output_dir: The directory to which to write model logs and output.
    mnist_location: The location of the MNIST numpy npz file.
    training_len: How long to run the model. A tuple of two values. The first
      value is the unit of measure (either "epochs" or "iterations") and the
      second is the number of units for which to train.
    masks: The masks, if any, used to prune weights. Masks can come in
      one of four forms:
      * A dictionary of numpy arrays. Each dictionary key is the name of the
        corresponding tensor that is to be masked out. Each value is a numpy
        array containing the masks (1 for including a weight, 0 for excluding).
      * The string name of a directory containing one file for each
        mask (in the form of foundations.save_restore).
      * A list of strings paths and dictionaries representing several masks.
        The mask used for training is the union of the pruned networks
        represented by these masks.
      * None, meaning the network should not be pruned.
    presets: The initial weights for the network, if any. Presets can come in
      any of the non-list forms mentioned for masks; each numpy array
      stores the desired initializations.
    train_order_seed: The random seed, if any, to be used to determine the
      order in which training examples are shuffled before being presented
      to the network.
  """
    # Retrieve previous information, if any.
    masks = save_restore.standardize(masks, union.union)
    presets = save_restore.standardize(presets)

    # Create the dataset and model.
    # todo: not working right now, problem with hyperparameters
    dataset = dataset_mnist.DatasetMnist(
        mnist_location, inc_dim=True, train_order_seed=train_order_seed)
    inputs, labels = dataset.placeholders
    model = model_conv2D.ModelConv2D(
        constants.HYPERPARAMETERS, inputs, labels, presets=presets, masks=masks)

    # Train.
    params = {'test_interval': 100, 'save_summaries': True, 'save_network': True}
    trainer.train(tf.Session(), dataset, model, constants.OPTIMIZER_FN,
                  training_len, output_dir, **params)