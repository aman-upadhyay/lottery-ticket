# Copyright (C) 2018 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A base class for all models to be used in lottery ticket experiments.

Defines a base class for a model that will be used for the lottery ticket
hypothesis experiment.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.nn import convolution
import numpy as np


class ModelBase(object):
    """A base class for all models used in lottery ticket experiments."""

    def __init__(self, presets=None, masks=None):
        """Creates dictionaries for storing references to model parameters.

    Args:
      presets: A dictionary mapping strings to numpy arrays. Each key is the
        name of a weight tensor; each value is the corresponding preset initial
        weights to which that tensor should be initialized.
      masks: A dictionary mapping strings to numpy arrays. Each key is the
        name of a weight tensor; each value is the corresponding mask (0 or 1
        values in each entry) that determines which weights have been pruned.
    """
        self._masks = masks if masks else {}
        self._presets = presets if presets else {}
        self._weights = {}

        self._train_summaries = None
        self._test_summaries = None
        self._validate_summaries = None

    @property
    def loss(self):
        return self._loss

    @property
    def train_summaries(self):
        return self._train_summaries

    @property
    def test_summaries(self):
        return self._test_summaries

    @property
    def validate_summaries(self):
        return self._validate_summaries

    @property
    def masks(self):
        return self._masks

    @property
    def presets(self):
        return self._presets

    @property
    def weights(self):
        return self._weights

    def get_current_weights(self, sess):
        output = {}
        for k, v in self.weights.items():
            output[k] = sess.run(v)
        return output

    def dense_layer(self,
                    name,
                    inputs,
                    units,
                    activation=None,
                    use_bias=True,
                    kernel_initializer=None):
        """Mimics tf.dense_layer but masks weights and uses presets as necessary."""
        # If there is a preset for this layer, use it.
        if name in self._presets:
            kernel_initializer = tf.constant_initializer(self._presets[name])

        # Create the weights.
        weights = tf.get_variable(
            name=name + '_w',
            shape=[inputs.shape[1], units],
            initializer=kernel_initializer)

        # Mask the layer as necessary.
        if name in self._masks:
            mask_initializer = tf.constant_initializer(self._masks[name])
            mask = tf.get_variable(
                name=name + '_m',
                shape=[inputs.shape[1], units],
                initializer=mask_initializer,
                trainable=False)
            weights = tf.multiply(weights, mask)

        self._weights[name] = weights

        # Compute the output.
        output = tf.matmul(inputs, weights)

        # Add bias if applicable.
        if use_bias:
            bias = tf.get_variable(
                name=name + '_b', shape=[units], initializer=tf.zeros_initializer())
            output += bias

        # Activate.
        if activation:
            return activation(output)
        else:
            return output

    def conv2D(self,
               name,
               inputs,
               kernel_size,
               filters,
               strides=1,
               activation=None,
               use_bias=True,
               kernel_initializer=None):
        """Mimics tf.Conv2D but masks weights and uses presets as necessary."""
        # If there is a preset for this layer, use it.
        if name in self._presets:
            kernel_initializer = tf.constant_initializer(self._presets[name])

        # Create the weights.
        weights = tf.get_variable(
            name=name + '_w',
            shape=[kernel_size[0], kernel_size[1], inputs.shape[3], filters],
            initializer=kernel_initializer)

        # Mask the layer as necessary.
        if name in self._masks:
            mask_initializer = tf.constant_initializer(self._masks[name])
            mask = tf.get_variable(
                name=name + '_m',
                shape=[kernel_size[0], kernel_size[1], inputs.shape[3], filters],
                initializer=mask_initializer,
                trainable=False)
            weights = tf.multiply(weights, mask)

        self._weights[name] = weights

        # Compute the output
        output = convolution(input=inputs, filter=weights, padding='VALID', strides=strides, dilation_rate=None, name=None,
                             data_format=None, filters=None, dilations=None)

        # Add bias if applicable.
        if use_bias:
            bias = tf.get_variable(
                name=name + '_b', shape=[filters], initializer=tf.zeros_initializer())
            output += bias

        # Activate.
        if activation:
            return activation(output)
        else:
            return output

    def dws_conv2D(self,
                   name,
                   inputs,
                   kernel_size,
                   filters,
                   strides=1,
                   activation=None,
                   use_bias=True,
                   kernel_initializer=None):
        """Mimics Depth-wise separable Conv2D but masks weights and uses presets as necessary."""
        # If there is a preset for this layer, use it.
        if name in self._presets:
            kernel_initializer = tf.constant_initializer(self._presets[name])

        # Create the weights. (Number of weights initialized more than required, this was done because of shape
        # constraints. But these extra weights dont participate in training so should not effect computation time.
        weights = tf.get_variable(
            name=name + '_w',
            shape=[kernel_size[0] + 1, kernel_size[1], inputs.shape[3], filters],
            initializer=kernel_initializer)

        # Mask the layer as necessary.
        if name in self._masks:
            mask_initializer = tf.constant_initializer(self._masks[name])
            mask = tf.get_variable(
                name=name + '_m',
                shape=[kernel_size[0] + 1, kernel_size[1], inputs.shape[3], filters],
                initializer=mask_initializer,
                trainable=False)
            weights = tf.multiply(weights, mask)

        self._weights[name] = weights

        # Compute the output
        sep = []
        fil = []
        for x in range(inputs.shape[3]):
            sep.append(convolution(
                input=tf.reshape(inputs[:, :, :, x], (tf.shape(inputs)[0], inputs.shape[1], inputs.shape[2], 1)),
                filter=tf.reshape(weights[:kernel_size[0], :, x, 0], (kernel_size[0], kernel_size[1], 1, 1)),
                padding='VALID', strides=1, dilation_rate=None, name=None, data_format=None, filters=None,
                dilations=None))
        stacked_sep = tf.stack(sep, axis=3)
        stacked_sep = tf.reshape(stacked_sep, (
            tf.shape(stacked_sep)[0], stacked_sep.shape[1], stacked_sep.shape[2], stacked_sep.shape[3]))
        for x in range(filters):
            fil.append(convolution(input=stacked_sep,
                                   filter=tf.reshape(weights[-1, -1, :, x], (1, 1, inputs.shape[3], 1)),
                                   padding='VALID', strides=1, dilation_rate=None, name=None, data_format=None,
                                   filters=None, dilations=None))
        output = tf.stack(fil, axis=3)
        output = tf.reshape(output, (tf.shape(output)[0], output.shape[1], output.shape[2], output.shape[3]))

        # Add bias if applicable.
        if use_bias:
            bias = tf.get_variable(
                name=name + '_b', shape=[filters], initializer=tf.zeros_initializer())
            output += bias

        # Activate.
        if activation:
            return activation(output)
        else:
            return output

    def dw_conv2D(self,
                  name,
                  inputs,
                  kernel_size,
                  filters=1,
                  strides=1,
                  activation=None,
                  use_bias=True,
                  kernel_initializer=None):
        """Mimics Depth-wise Conv2D but masks weights and uses presets as necessary."""
        # If there is a preset for this layer, use it.
        if name in self._presets:
            kernel_initializer = tf.constant_initializer(self._presets[name])

        weights = tf.get_variable(
            name=name + '_w',
            shape=[kernel_size[0], kernel_size[1], inputs.shape[3], filters],
            initializer=kernel_initializer)

        # Mask the layer as necessary.
        if name in self._masks:
            mask_initializer = tf.constant_initializer(self._masks[name])
            mask = tf.get_variable(
                name=name + '_m',
                shape=[kernel_size[0], kernel_size[1], inputs.shape[3], filters],
                initializer=mask_initializer,
                trainable=False)
            weights = tf.multiply(weights, mask)

        self._weights[name] = weights

        # Compute the output
        sep = []
        for x in range(inputs.shape[3]):
            sep.append(convolution(
                input=tf.reshape(inputs[:, :, :, x], (tf.shape(inputs)[0], inputs.shape[1], inputs.shape[2], 1)),
                filter=tf.reshape(weights[:, :, x, :], (kernel_size[0], kernel_size[1], 1, 1)),
                padding='VALID', strides=strides, dilation_rate=None, name=None, data_format=None, filters=None,
                dilations=None))
        stacked_sep = tf.stack(sep, axis=3)
        stacked_sep = tf.reshape(stacked_sep, (
            tf.shape(stacked_sep)[0], stacked_sep.shape[1], stacked_sep.shape[2], stacked_sep.shape[3]))
        output = stacked_sep

        # Add bias if applicable.
        if use_bias:
            bias = tf.get_variable(
                name=name + '_b', shape=[filters], initializer=tf.zeros_initializer())
            output += bias

        # Activate.
        if activation:
            return activation(output)
        else:
            return output

    def flatten(self,
                name,
                inputs):
        """Mimics tf.Flatten"""

        # Create the weights.
        weights = tf.get_variable(
            name=name + '_w',
            shape=[0],
            initializer=None)

        # Mask the layer as necessary.
        if name in self._masks:
            mask_initializer = tf.constant_initializer(self._masks[name])
            mask = tf.get_variable(
                name=name + '_m',
                shape=[0],
                initializer=mask_initializer,
                trainable=False)
            weights = tf.multiply(weights, mask)

        self._weights[name] = weights

        output = tf.reshape(inputs, [tf.shape(inputs)[0], inputs.shape[1] * inputs.shape[2] * inputs.shape[3]])

        return output

    def create_loss_and_accuracy(self, label_placeholder, output_logits):
        """Creates loss and accuracy once a child class has created the network."""
        # Compute cross-entropy loss and accuracy.
        self._loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=label_placeholder, logits=output_logits))
        accuracy = tf.reduce_mean(
            tf.cast(
                tf.equal(
                    tf.argmax(label_placeholder, 1),
                    tf.argmax(tf.nn.softmax(output_logits), 1)), tf.float32))

        # Create summaries for loss and accuracy.
        self._train_summaries = [
            tf.summary.scalar('train_loss', self._loss),
            tf.summary.scalar('train_accuracy', accuracy)
        ]
        self._test_summaries = [
            tf.summary.scalar('test_loss', self._loss),
            tf.summary.scalar('test_accuracy', accuracy)
        ]
        self._validate_summaries = [
            tf.summary.scalar('validate_loss', self._loss),
            tf.summary.scalar('validate_accuracy', accuracy)
        ]
