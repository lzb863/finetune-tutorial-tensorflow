from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import re
import tensorflow as tf
from models.deeplab_v3 import DeepLabv3

train_layers = ['deeplab_v3']


class Model(object):
    def __init__(self, sess, config):
        self.sess = sess
        self.config = config

        self.init_global_step()
        self.init_cur_epoch()
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)

        image_dim = [self.config.input_height, self.config.input_width, 3]
        label_dim = [self.config.input_height, self.config.input_width, 1]

        self.images = tf.placeholder(tf.float32, shape=[None] + image_dim, name='images')
        self.labels = tf.placeholder(tf.int32, shape=[None] + label_dim, name='labels')

        net = DeepLabv3(self.sess, self.config, self.is_training)

        self.logits = net(self.images)

        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.squeeze(self.labels, axis=3), logits=self.logits,
                                                           name="cross_entropy"))

        # List of trainable variables of the layers we want to train
        var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

        gradients = tf.gradients(self.loss, var_list)
        gradients = list(zip(gradients, var_list))

        self.learning_rate = tf.train.polynomial_decay(self.config.learning_rate, self.cur_epoch_tensor,
                                                       self.config.num_epochs, end_learning_rate=1e-06, power=0.9)
        optimizer = tf.train.AdamOptimizer(self.learning_rate, name='Adam')
        self.train_step = optimizer.apply_gradients(grads_and_vars=gradients)

    def init_global_step(self):
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
            self.increment_cur_step_tensor = tf.assign(self.global_step_tensor, self.global_step_tensor + 1)

    def init_cur_epoch(self):
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor + 1)

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=1)

    @property
    def model_dir(self):
        return "{}_{}".format(self.config.dataset, self.config.num_epochs)

    def save(self, sess):
        print("Saving model...")
        model_name = self.config.model + ".model"
        checkpoint_dir = os.path.join(self.config.checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(sess, os.path.join(checkpoint_dir, model_name), self.global_step_tensor)

        print("Model saved")

    def load(self, sess):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(self.config.checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            steps = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))

            return True, steps
        else:
            print(" [*] Failed to find a checkpoint")

            return False, 0
