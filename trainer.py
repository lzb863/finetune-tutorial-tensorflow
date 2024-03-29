from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import time
import tensorflow as tf


class Trainer(object):
    def __init__(self, sess, model, train_data_loader, valid_data_loader, config):
        self.sess = sess
        self.model = model
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.config = config

        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)

    def train(self):
        self.start_time = time.time()
        could_load, checkpoint_counter = self.model.load(self.sess)
        if could_load:
            self.sess.run(tf.assign(self.model.global_step_tensor, checkpoint_counter))
            print(" [*] Load success!")
        else:
            print(" [!] Load failed...")

        self.num_iter_per_epoch = self.train_data_loader.num_samples // self.config.batch_size

        self.batch_train_images, self.batch_train_labels = self.train_data_loader.next_batch(self.config.batch_size)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

        for cur_epoch in xrange(self.model.cur_epoch_tensor.eval(self.sess), self.config.num_epochs, 1):
            self.train_epoch()
            self.sess.run(self.model.increment_cur_epoch_tensor)

        coord.request_stop()
        coord.join(threads)

    def train_epoch(self):
        for cur_iter in xrange(0, self.num_iter_per_epoch):
            self.train_step(cur_iter)

        self.model.save(self.sess)

    def train_step(self, iter):
        batch_images, batch_labels = self.sess.run([self.batch_train_images, self.batch_train_labels])
        feed_dict = {self.model.images: batch_images, self.model.labels: batch_labels, self.model.is_training: True}

        _, loss = self.sess.run([self.model.train_step, self.model.loss], feed_dict=feed_dict)

        self.sess.run(self.model.increment_cur_step_tensor)

        print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, train loss: %.8f" % (
            self.model.cur_epoch_tensor.eval(self.sess), self.config.num_epochs, iter, self.num_iter_per_epoch,
            time.time() - self.start_time, loss))

    def test(self):
        load_model_status, global_steps = self.model.load(self.sess)
        assert load_model_status == True, " [!] Load weights FAILED..."
        print(" [*] Load weights SUCCESS...")

        self.single_test_image, self.single_test_label = self.valid_data_loader.next_sample()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

        num_test_samples = self.valid_data_loader.num_samples

        for idx in xrange(num_test_samples):
            single_image, single_label = self.sess.run([self.single_test_image, self.single_test_label])
            feed_dict = {self.model.images: single_image, self.model.labels: single_label,
                         self.model.is_training: False}
            loss = self.sess.run(self.model.loss, feed_dict=feed_dict)

            print("Sample %d loss: %.8f" % (idx + 1, loss))

        coord.request_stop()
        coord.join(threads)
