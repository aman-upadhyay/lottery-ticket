"""A Conv2D neural network model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from foundations import model_base
import tensorflow as tf


class ModelConv2D(model_base.ModelBase):
  """A Conv2D network with user-specifiable hyperparameters."""

  def __init__(self,
               hyperparameters,
               input_placeholder,
               label_placeholder,
               presets=None,
               masks=None):
    """Creates a Conv2D network.

    Args:
      hyperparameters: A dictionary of hyperparameters for the network.
        For this class, a single hyperparameter is available: 'layers'.
        This key's value is a list of (layer_id 2 for conv2D and 1 for dense,
        kernel_size/ 0 for dense, filters/units for dense, strides/0 for
        dense, activation function) tuples for each layer in order from input
        to output. If the activation function is None, then no activation will
        be used. For flatten layer all 5 values of the list is 0.
      input_placeholder: A placeholder for the network's input.
      label_placeholder: A placeholder for the network's expected output.
      presets: Preset initializations for the network as in model_base.py
      masks: Masks to prune the network as in model_base.py.
    """
    # Call parent constructor.
    super(ModelConv2D, self).__init__(presets=presets, masks=masks)

    # Build the network layer by layer.
    current_layer = input_placeholder
    for i, (layer_id, kernel_size, filters, strides, activation) in enumerate(hyperparameters['layers']):
        if layer_id == 2:
            current_layer = self.conv2D(
                'layer{}'.format(i),
                current_layer,
                kernel_size,
                filters,
                strides,
                activation,
                kernel_initializer=tf.contrib.layers.xavier_initializer(
                    uniform=False))
        elif layer_id == 1:
            current_layer = self.dense_layer(
                'layer{}'.format(i),
                current_layer,
                filters,
                activation,
                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        else:
            current_layer = self.flatten(
                'layer{}'.format(i),
                current_layer)

    # Compute the loss and accuracy.
    self.create_loss_and_accuracy(label_placeholder, current_layer)
