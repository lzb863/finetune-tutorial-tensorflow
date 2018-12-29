from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from ops import ConvBlock


class ASPP(object):
    '''
    AtrousSpatialPyramidPoolingModule: consists of (a) one 1x1 convolution and three 3x3 convolutions with
    rates = (6, 12, 18) when output stride = 16 (all with 256 filters and batch normalization)
    '''

    def __init__(self, depth=256, is_training=True):
        self.avg_pool_conv = ConvBlock(depth, 1, is_training=is_training, name='avg_pool')

        self.atrous_pool_block_1 = ConvBlock(depth, 1, is_training=is_training, name='block1')
        self.atrous_pool_block_6 = ConvBlock(depth, 3, dilation=6, is_training=is_training, name='block2')
        self.atrous_pool_block_12 = ConvBlock(depth, 3, dilation=12, is_training=is_training, name='block3')
        self.atrous_pool_block_18 = ConvBlock(depth, 3, dilation=18, is_training=is_training, name='block4')

        self.conv_out = ConvBlock(depth, 1, is_training=is_training, name='conv1')

    def forward(self, inputs):
        with tf.variable_scope('aspp'):
            feature_map_size = tf.shape(inputs)

            # Global average pooling
            image_features = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
            image_features = self.avg_pool_conv.forward(image_features)
            image_features = tf.image.resize_bilinear(image_features, (feature_map_size[1], feature_map_size[2]))

            out_1x1_1 = self.atrous_pool_block_1.forward(inputs)
            out_3x3_6 = self.atrous_pool_block_6.forward(inputs)
            out_3x3_12 = self.atrous_pool_block_12.forward(inputs)
            out_3x3_18 = self.atrous_pool_block_18.forward(inputs)

            out = tf.concat([image_features, out_1x1_1, out_3x3_6, out_3x3_12, out_3x3_18], axis=3)
            out = self.conv_out.forward(out)

            return out

    def __call__(self, x):
        return self.forward(x)
