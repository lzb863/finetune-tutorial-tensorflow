from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def conv(x, out_channels, kernel_size, stride, padding='SAME', bias=False):
    """Reference: https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html"""
    input_channels = int(x.get_shape()[-1])
    convolve = lambda i, k: tf.nn.conv2d(i, k, strides=[1, stride, stride, 1], padding=padding)

    weights = tf.get_variable('weights', shape=[kernel_size, kernel_size, input_channels, out_channels],
                              initializer=tf.truncated_normal_initializer(stddev=0.02))
    if bias:
        biases = tf.get_variable('biases', shape=[out_channels])

    out = convolve(x, weights)
    if bias:
        out = tf.reshape(tf.nn.bias_add(out, biases), out.get_shape().as_list())

    return out


class ConvBlock(object):
    def __init__(self, out_channels, kernel_size=3, stride=1, padding="SAME", dilation=1, bias=False, use_bn=False,
                 use_act=False, is_training=True, name=None):
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        self.use_bn = use_bn
        self.use_act = use_act
        self.is_training = is_training
        self.name = name

    def forward(self, x):
        with tf.variable_scope(self.name):
            out = conv(x, self.out_channels, self.kernel_size, self.stride, self.padding, self.bias)

            if self.use_bn:
                out = tf.layers.batch_normalization(out, training=self.is_training, name='BatchNorm')

            if self.use_act:
                out = tf.nn.relu(out)

            return out

    def __call__(self, x):
        return self.forward(x)


class MaxPool2d(object):
    def __init__(self, kernel_size, stride=2, padding="valid"):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return tf.layers.max_pooling2d(x, self.kernel_size, self.stride, self.padding)

    def __call__(self, x):
        return self.forward(x)


class AvgPool2d(object):
    def __init__(self, kernel_size, stride=2, padding="valid"):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return tf.layers.average_pooling2d(x, self.kernel_size, self.stride, self.padding)

    def __call__(self, x):
        return self.forward(x)


class FullyConnected(object):
    def __init__(self, out_channels):
        self.out_channels = out_channels

    def forward(self, x):
        return tf.layers.dense(x, self.out_channels)

    def __call__(self, x):
        return self.forward(x)


class Sequential(object):
    def __init__(self, layers, scope=None):
        self.layers = layers
        self.scope = scope

    def forward(self, x):
        with tf.variable_scope(self.scope):
            out = x
            for i, layer in enumerate(self.layers):
                with tf.variable_scope('unit_{}'.format(i + 1)):
                    out = layer(out)

            return out

    def __call__(self, x):
        return self.forward(x)
