"""MobileNet model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from foundations import model_base
import tensorflow as tf


class MobileNet(model_base.ModelBase):
    """MobileNet model."""

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
        super(MobileNet, self).__init__(presets=presets, masks=masks)

        # Build the network layer by layer.
        current_layer = input_placeholder
        current_layer = self.conv2D(
            'layer0',
            current_layer,
            (3, 3),
            32,
            strides=2,
            activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer(
                uniform=False))
        current_layer = tf.keras.layers.BatchNormalization()(current_layer)
        current_layer = tf.nn.relu(current_layer)
        current_layer = self.dw_conv2D(
            'layer1',
            current_layer,
            kernel_size=(3,3),
            activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer(
                uniform=False))
        current_layer = tf.keras.layers.BatchNormalization()(current_layer)
        current_layer = tf.nn.relu(current_layer)
        current_layer = self.conv2D(
            'layer2',
            current_layer,
            (1,1),
            64,
            strides=1,
            activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer(
                uniform=False))
        current_layer = tf.keras.layers.BatchNormalization()(current_layer)
        current_layer = tf.nn.relu(current_layer)
        current_layer = self.dw_conv2D(
            'layer3',
            current_layer,
            kernel_size=(3, 3),
            strides=2,
            activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer(
                uniform=False))
        current_layer = tf.keras.layers.BatchNormalization()(current_layer)
        current_layer = tf.nn.relu(current_layer)
        current_layer = self.conv2D(
            'layer4',
            current_layer,
            (1, 1),
            128,
            strides=1,
            activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer(
                uniform=False))
        current_layer = tf.keras.layers.BatchNormalization()(current_layer)
        current_layer = tf.nn.relu(current_layer)
        current_layer = self.dw_conv2D(
            'layer5',
            current_layer,
            kernel_size=(3, 3),
            strides=1,
            activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer(
                uniform=False))
        current_layer = tf.keras.layers.BatchNormalization()(current_layer)
        current_layer = tf.nn.relu(current_layer)
        current_layer = self.conv2D(
            'layer6',
            current_layer,
            (1, 1),
            128,
            strides=1,
            activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer(
                uniform=False))
        current_layer = tf.keras.layers.BatchNormalization()(current_layer)
        current_layer = tf.nn.relu(current_layer)
        current_layer = self.dw_conv2D(
            'layer7',
            current_layer,
            kernel_size=(3, 3),
            strides=2,
            activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer(
                uniform=False))
        current_layer = tf.keras.layers.BatchNormalization()(current_layer)
        current_layer = tf.nn.relu(current_layer)
        current_layer = self.conv2D(
            'layer8',
            current_layer,
            (1, 1),
            256,
            strides=1,
            activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer(
                uniform=False))
        current_layer = tf.keras.layers.BatchNormalization()(current_layer)
        current_layer = tf.nn.relu(current_layer)
        current_layer = self.dw_conv2D(
            'layer9',
            current_layer,
            kernel_size=(3, 3),
            strides=1,
            activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer(
                uniform=False))
        current_layer = tf.keras.layers.BatchNormalization()(current_layer)
        current_layer = tf.nn.relu(current_layer)
        current_layer = self.conv2D(
            'layer10',
            current_layer,
            (1, 1),
            256,
            strides=1,
            activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer(
                uniform=False))
        current_layer = tf.keras.layers.BatchNormalization()(current_layer)
        current_layer = tf.nn.relu(current_layer)
        current_layer = self.dw_conv2D(
            'layer11',
            current_layer,
            kernel_size=(3, 3),
            strides=2,
            activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer(
                uniform=False))
        current_layer = tf.keras.layers.BatchNormalization()(current_layer)
        current_layer = tf.nn.relu(current_layer)
        current_layer = self.conv2D(
            'layer12',
            current_layer,
            (1, 1),
            512,
            strides=1,
            activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer(
                uniform=False))
        current_layer = tf.keras.layers.BatchNormalization()(current_layer)
        current_layer = tf.nn.relu(current_layer)
        current_layer = self.dw_conv2D(
            'layer13',
            current_layer,
            kernel_size=(3, 3),
            activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer(
                uniform=False))
        current_layer = tf.keras.layers.BatchNormalization()(current_layer)
        current_layer = tf.nn.relu(current_layer)
        current_layer = self.conv2D(
            'layer14',
            current_layer,
            (1, 1),
            64,
            strides=1,
            activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer(
                uniform=False))
        current_layer = tf.keras.layers.BatchNormalization()(current_layer)
        current_layer = tf.nn.relu(current_layer)
        current_layer = self.dw_conv2D(
            'layer15',
            current_layer,
            kernel_size=(3, 3),
            activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer(
                uniform=False))
        current_layer = tf.keras.layers.BatchNormalization()(current_layer)
        current_layer = tf.nn.relu(current_layer)
        current_layer = self.conv2D(
            'layer16',
            current_layer,
            (1, 1),
            64,
            strides=1,
            activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer(
                uniform=False))
        current_layer = tf.keras.layers.BatchNormalization()(current_layer)
        current_layer = tf.nn.relu(current_layer)
        current_layer = self.dw_conv2D(
            'layer17',
            current_layer,
            kernel_size=(3,3),
            activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer(
                uniform=False))
        current_layer = tf.keras.layers.BatchNormalization()(current_layer)
        current_layer = tf.nn.relu(current_layer)
        current_layer = self.conv2D(
            'layer18',
            current_layer,
            (1,1),
            64,
            strides=1,
            activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer(
                uniform=False))
        current_layer = tf.keras.layers.BatchNormalization()(current_layer)
        current_layer = tf.nn.relu(current_layer)
        current_layer = self.dw_conv2D(
            'layer19',
            current_layer,
            kernel_size=(3,3),
            activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer(
                uniform=False))
        current_layer = tf.keras.layers.BatchNormalization()(current_layer)
        current_layer = tf.nn.relu(current_layer)
        current_layer = self.conv2D(
            'layer20',
            current_layer,
            (1,1),
            64,
            strides=1,
            activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer(
                uniform=False))
        current_layer = tf.keras.layers.BatchNormalization()(current_layer)
        current_layer = tf.nn.relu(current_layer)
        current_layer = self.dw_conv2D(
            'layer21',
            current_layer,
            kernel_size=(3,3),
            activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer(
                uniform=False))
        current_layer = tf.keras.layers.BatchNormalization()(current_layer)
        current_layer = tf.nn.relu(current_layer)
        current_layer = self.conv2D(
            'layer22',
            current_layer,
            (1,1),
            64,
            strides=1,
            activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer(
                uniform=False))
        current_layer = tf.keras.layers.BatchNormalization()(current_layer)
        current_layer = tf.nn.relu(current_layer)
        current_layer = self.dw_conv2D(
            'layer25',
            current_layer,
            kernel_size=(3, 3),
            strides=2,
            activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer(
                uniform=False))
        current_layer = tf.keras.layers.BatchNormalization()(current_layer)
        current_layer = tf.nn.relu(current_layer)
        current_layer = self.conv2D(
            'layer26',
            current_layer,
            (1, 1),
            64,
            strides=1,
            activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer(
                uniform=False))
        current_layer = tf.keras.layers.BatchNormalization()(current_layer)
        current_layer = tf.nn.relu(current_layer)
        current_layer = tf.keras.layers.AveragePooling2D(pool_size=(7, 7))(current_layer)
        current_layer = tf.keras.layers.Flatten()(current_layer)
        current_layer = self.dense_layer(
            'layer27',
            current_layer,
            1000,
            None,
            kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        current_layer = self.dense_layer(
            'layer28',
            current_layer,
            10,
            tf.nn.softmax,
            kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))

        # Compute the loss and accuracy.
        self.create_loss_and_accuracy(label_placeholder, current_layer)
