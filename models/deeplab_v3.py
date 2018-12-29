from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import resnet
from aspp import ASPP
from ops import ConvBlock


class DeepLabv3(object):
    def __init__(self, sess, config, is_training=True):
        self.resnet_model = resnet.__dict__.get(config.resnet_model)(sess, pretrained=True, is_training=is_training)
        self.aspp = ASPP(is_training=is_training)
        self.conv = ConvBlock(config.num_classes, 1, 1, is_training=is_training, name='conv1')

    def forward(self, x):
        input_shape = x.get_shape().as_list()
        _, end_points = self.resnet_model(x)

        with tf.variable_scope('deeplab_v3'):
            out = end_points[-1]
            out = self.aspp(out)
            out = self.conv(out)

            out = tf.image.resize_bilinear(out, (input_shape[1], input_shape[2]))

            return out

    def __call__(self, x):
        return self.forward(x)
