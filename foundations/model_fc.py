"""A fully-connected neural network model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from foundations import model_base
import tensorflow as tf


class ModelFc(model_base.ModelBase):
  """A fully-connected network with user-specifiable hyperparameters."""

  def __init__(self,
               hyperparameters,
               input_placeholder,
               label_placeholder,
               presets=None,
               masks=None):
    """Creates a fully-connected network.

    Args:
      hyperparameters: A dictionary of hyperparameters for the network.
        For this class, a single hyperparameter is available: 'layers'. This
        key's value is a list of (# of units, activation function) tuples
        for each layer in order from input to output. If the activation
        function is None, then no activation will be used.
      input_placeholder: A placeholder for the network's input.
      label_placeholder: A placeholder for the network's expected output.
      presets: Preset initializations for the network as in model_base.py
      masks: Masks to prune the network as in model_base.py.
    """
    # Call parent constructor.
    super(ModelFc, self).__init__(presets=presets, masks=masks)

    # Build the network layer by layer.
    current_layer = input_placeholder
    for i, (units, activation) in enumerate(hyperparameters['layers']):
      current_layer = self.dense_layer(
          'layer{}'.format(i),
          current_layer,
          units,
          activation,
          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))

    # Compute the loss and accuracy.
    self.create_loss_and_accuracy(label_placeholder, current_layer)
