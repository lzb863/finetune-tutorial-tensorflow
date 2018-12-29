from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from ops import ConvBlock, MaxPool2d, AvgPool2d, FullyConnected, Sequential

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

pretrained_model_path = '/home/tramac/PycharmProjects/finetune-tutorial-tensorflow/pretrained_models/resnet_v2_50/resnet_v2_50.ckpt'


class BasicBlock(object):
    expansion = 1

    def __init__(self, planes, stride=1, downsample=None, is_training=True):
        self.conv_block1 = ConvBlock(planes, stride=stride, use_bn=True, use_act=True, is_training=is_training)
        self.conv_block2 = ConvBlock(planes, use_bn=True, use_act=False, is_training=is_training)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv_block1(x)
        out = self.conv_block2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = tf.nn.relu(out)

        return out

    def __call__(self, x):
        return self.forward(x)


class Bottleneck(object):
    expansion = 4

    def __init__(self, planes, stride=1, downsample=None, is_training=True):
        self.conv_block1 = ConvBlock(planes, 1, use_bn=True, use_act=True, is_training=is_training, name='conv1')
        self.conv_block2 = ConvBlock(planes, stride=stride, use_bn=True, use_act=True, is_training=is_training,
                                     name='conv2')
        self.conv_block3 = ConvBlock(planes * self.expansion, 1, use_bn=True, use_act=False, is_training=is_training,
                                     name='conv3')
        self.downsample = downsample

    def forward(self, x):
        with tf.variable_scope('bottleneck_v2'):
            identity = x

            out = self.conv_block1(x)
            out = self.conv_block2(out)
            out = self.conv_block3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            out = tf.nn.relu(out)

            return out

    def __call__(self, x):
        return self.forward(x)


class ResNet(object):
    def __init__(self, block, layers, num_classes=1000, is_training=True, scope=None):
        self.scope = scope

        self.inplanes = 64
        self.conv_block1 = ConvBlock(64, 7, 2, use_bn=True, use_act=True, is_training=is_training, name='conv1')
        self.maxpool = MaxPool2d(3, 2, "same")
        self.layer1 = self._make_layer(block, 64, layers[0], is_training=is_training, name='block1')
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, is_training=is_training, name='block2')
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, is_training=is_training, name='block3')
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, is_training=is_training, name='block4')
        self.avgpool = AvgPool2d(7, 7)
        self.fc = FullyConnected(num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, is_training=True, name=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = ConvBlock(planes * block.expansion, 1, stride, use_bn=True, is_training=is_training,
                                   name='shortcut')

        layers = []
        layers.append(block(planes, stride, downsample, is_training))
        self.inplanes = planes * block.expansion
        for _ in xrange(1, blocks):
            layers.append(block(planes))

        return Sequential(layers, name)

    def forward(self, x):
        end_points = []
        with tf.variable_scope(self.scope):
            x = self.conv_block1(x)
            end_points.append(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            end_points.append(x)
            x = self.layer2(x)
            end_points.append(x)
            x = self.layer3(x)
            end_points.append(x)
            x = self.layer4(x)
            end_points.append(x)

            x = self.avgpool(x)
            x = tf.layers.flatten(x)
            x = self.fc(x)

            return x, end_points

    def load_initial_weights(self, sess):
        # version 1
        '''
        reader = pywrap_tensorflow.NewCheckpointReader('../pretrained_models/resnet_v2_50/resnet_v2_50.ckpt')
        weights_dict = reader.get_variable_to_shape_map()

        for op_name in weights_dict:
            with tf.variable_scope('resnet_v2_50', reuse=True):
                var = tf.get_variable(op_name[13:], trainable=False)
                value = reader.get_tensor(op_name)
                sess.run(var.assign(value))
        '''

        # version 2
        reader = pywrap_tensorflow.NewCheckpointReader(pretrained_model_path)
        resnet_variable_name = [v.name for v in tf.trainable_variables() if 'resnet' in v.name]
        for name in resnet_variable_name:
            with tf.variable_scope('resnet_v2_50', reuse=True):
                # version 2.1
                '''
                var = tf.get_variable(name[13:-2], trainable=False)
                value = reader.get_tensor(name[:-2])
                sess.run(var.assign(value))
                '''

                # version 2.2
                try:
                    var = tf.get_variable(name[13:-2], trainable=False)
                    value = reader.get_tensor(name[:-2])
                    sess.run(var.assign(value))
                except:
                    continue

    def __call__(self, x):
        return self.forward(x)


def resnet18(sess, pretrained=False, is_training=True, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], is_training=is_training, scope='resnet_v2_18', **kwargs)
    if pretrained:
        model.load_initial_weights(sess)
    return model


def resnet34(sess, pretrained=False, is_training=True, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], is_training=is_training, scope='resnet_v2_34', **kwargs)
    if pretrained:
        model.load_initial_weights(sess)
    return model


def resnet50(sess, pretrained=False, is_training=True, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], is_training=is_training, scope='resnet_v2_50', **kwargs)
    if pretrained:
        model.load_initial_weights(sess)
    return model


def resnet101(sess, pretrained=False, is_training=True, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], is_training=is_training, scope='resnet_v2_101', **kwargs)
    if pretrained:
        model.load_initial_weights(sess)
    return model


def resnet152(sess, pretrained=False, is_training=True, **kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], is_training=is_training, scope='resnet_v2_152', **kwargs)
    if pretrained:
        model.load_initial_weights(sess)
    return model
