"""ResNet50 model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from foundations import model_base
import tensorflow as tf


class ResNet50(model_base.ModelBase):
    """ResNet50 model."""

    def __init__(self,
                 input_placeholder,
                 label_placeholder,
                 presets=None,
                 masks=None):
        """Creates a ResNet50 network.

    Args:
      input_placeholder: A placeholder for the network's input.
      label_placeholder: A placeholder for the network's expected output.
      presets: Preset initializations for the network as in model_base.py
      masks: Masks to prune the network as in model_base.py.
    """
        # Call parent constructor.
        super(ResNet50, self).__init__(presets=presets, masks=masks)

        # clubbing some layers for future use
        def res_identity(x, filters, i):
            # renet block where dimension doesnot change.
            # The skip connection is just simple identity conncection
            # we will have 3 blocks and then input will be added

            x_skip = x  # this will be used for addition with the residual block
            f1, f2 = filters

            # first block
            x = self.conv2D('layer{}'.format(str(i + 1)), x, filters=f1, kernel_size=(1, 1), strides=(1, 1),
                            pad='VALID',
                            kernel_initializer=tf.contrib.layers.xavier_initializer(
                                uniform=False), activation=None)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.nn.relu(x)

            # second block # bottleneck (but size kept same with padding)
            x = self.conv2D('layer{}'.format(str(i + 2)), x, filters=f1, kernel_size=(3, 3), strides=(1, 1), pad='SAME',
                            kernel_initializer=tf.contrib.layers.xavier_initializer(
                                uniform=False), activation=None)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.nn.relu(x)

            # third block activation used after adding the input
            x = self.conv2D('layer{}'.format(str(i + 3)), x, filters=f2, kernel_size=(1, 1), strides=(1, 1),
                            pad='VALID',
                            kernel_initializer=tf.contrib.layers.xavier_initializer(
                                uniform=False), activation=None)
            x = tf.keras.layers.BatchNormalization()(x)
            # x = tf.nn.relu(x)

            # add the input
            x = tf.keras.layers.Add()([x, x_skip])
            x = tf.nn.relu(x)

            return x

        def res_conv(x, s, filters, i):
            '''
            here the input size changes'''
            x_skip = x
            f1, f2 = filters

            # first block
            x = self.conv2D('layer{}'.format(str(i + 1)), x, filters=f1, kernel_size=(1, 1), strides=(s, s),
                            pad='VALID',
                            kernel_initializer=tf.contrib.layers.xavier_initializer(
                                uniform=False), activation=None)
            # when s = 2 then it is like downsizing the feature map
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.nn.relu(x)

            # second block
            x = self.conv2D('layer{}'.format(str(i + 2)), x, filters=f1, kernel_size=(3, 3), strides=(1, 1), pad='SAME',
                            kernel_initializer=tf.contrib.layers.xavier_initializer(
                                uniform=False), activation=None)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.nn.relu(x)

            # third block
            x = self.conv2D('layer{}'.format(str(i + 3)), x, filters=f2, kernel_size=(1, 1), strides=(1, 1),
                            pad='VALID',
                            kernel_initializer=tf.contrib.layers.xavier_initializer(
                                uniform=False), activation=None)
            x = tf.keras.layers.BatchNormalization()(x)

            # shortcut
            x_skip = self.conv2D('layer{}'.format(str(i + 4)), x_skip, filters=f2, kernel_size=(1, 1), strides=(s, s),
                                 pad='VALID',
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(
                                     uniform=False), activation=None)
            x_skip = tf.keras.layers.BatchNormalization()(x_skip)

            # add
            x = tf.keras.layers.Add()([x, x_skip])
            x = tf.nn.relu(x)

            return x

        # Build the network layer by layer.
        current_layer = input_placeholder
        current_layer = tf.keras.layers.ZeroPadding2D(padding=(3, 3))(current_layer)
        current_layer = self.conv2D('layer0',
                                    current_layer,
                                    kernel_size=(7, 7),
                                    strides=2,
                                    filters=64,
                                    activation=None,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        current_layer = tf.keras.layers.BatchNormalization()(current_layer)
        current_layer = tf.nn.relu(current_layer)
        current_layer = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(current_layer)
        current_layer = res_conv(current_layer, s=1, filters=(64, 256), i=0)
        current_layer = res_identity(current_layer, filters=(64, 256), i=4)
        current_layer = res_identity(current_layer, filters=(64, 256), i=7)
        current_layer = res_conv(current_layer, s=2, filters=(128, 512), i=10)
        current_layer = res_identity(current_layer, filters=(128, 512), i=14)
        current_layer = res_identity(current_layer, filters=(128, 512), i=17)
        current_layer = res_identity(current_layer, filters=(128, 512), i=20)
        current_layer = res_conv(current_layer, s=2, filters=(256, 1024), i=23)
        current_layer = res_identity(current_layer, filters=(256, 1024), i=27)
        current_layer = res_identity(current_layer, filters=(256, 1024), i=30)
        current_layer = res_identity(current_layer, filters=(256, 1024), i=33)
        current_layer = res_identity(current_layer, filters=(256, 1024), i=36)
        current_layer = res_identity(current_layer, filters=(256, 1024), i=39)
        current_layer = res_conv(current_layer, s=2, filters=(512, 2048), i=42)
        current_layer = res_identity(current_layer, filters=(512, 2048), i=46)
        current_layer = res_identity(current_layer, filters=(512, 2048), i=49)
        current_layer = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(current_layer)

        current_layer = tf.keras.layers.Flatten()(current_layer)
        current_layer = self.dense_layer('layer53', current_layer, 100, activation=tf.nn.softmax,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(
                                             uniform=False))  # multi-class
        # Compute the loss and accuracy.
        self.create_loss_and_accuracy(label_placeholder, current_layer)
