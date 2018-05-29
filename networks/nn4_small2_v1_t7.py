#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 ccmicky <chengchun@imdada.cn>
# Licensed under the Dada tech.co.ltd - http://www.imdada.cn

import tensorflow as tf
from networks.network import Network


class nn4_small2_v1_v7(Network):
    ontrain = False
    def __init__(self, trainable=True, ontrain=True):
        self.inputs = []
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3])
        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = dict({'data': self.data, 'im_info': self.im_info})
        self.trainable = trainable
        self.setup()
        self.ontrain = ontrain

        with tf.variable_scope('fc_face_vec', reuse=True):
            weights = tf.get_variable("weights")
            biases = tf.get_variable("biases")
            self.vec_weights = weights
            self.vec_bias = biases


    def setup(self):

        (self.feed('data')
         .conv(7, 7, 64, 2, 2, name='nn4_conv1_1', trainable=True, batch_norm=True, ontrain=self.ontrain)
         .max_pool(3, 3, 2, 2, padding='SAME', name='nn4_pool1')
         .lrn(5, 0.0001, 0.75, name='nn4_lrn_1_1')
         .conv(1, 1, 64, 1, 1, name='nn4_conv1_2', trainable=True, batch_norm=True, ontrain=self.ontrain)
         .conv(3, 3, 192, 1, 1, name='nn4_conv2_1', trainable=True, batch_norm=True, ontrain=self.ontrain)
         .lrn(5, 0.0001, 0.75, name='nn4_lrn_2_1')
         .max_pool(3, 3, 2, 2, padding='SAME', name='nn4_pool2'))

        (self.feed('nn4_pool2')
         .Inception((3, 5), (1, 1), (128, 32), (96, 16, 32, 64), 3, 3, 2, 2,
                    name="Inception_1", ontrain=self.ontrain)
         .Inception((3, 5), (1, 1), (128, 64), (96, 32, 64, 64), 3, 3, 3, 3,
                    name="Inception_2", ontrain=self.ontrain)
         .Inception((3, 5), (2, 2), (256, 64), (128, 32, None, None), 3, 3, 2,
                    2, name="Inception_3", ontrain=self.ontrain)
         .Inception((3, 5), (1, 1), (192, 64), (96, 32, 128, 256), 3, 3, 3, 3,
                    name="Inception_4", ontrain=self.ontrain)
         .Inception((3, 5), (2, 2), (256, 128), (160, 64, None, None), 3, 3, 2,
                    2, name="Inception_5", ontrain=self.ontrain)
         .Inception((3,), (1,), (384,), (96, 96, 256), 3, 3, 3, 3,
                    name="Inception_6", ontrain=self.ontrain)
         .Inception((3,), (1,), (384,), (96, 96, 256), 3, 3, 2, 2,
                    name="Inception_7", ontrain=self.ontrain))

        (self.feed('Inception_7')
         .avg_pool(3, 3, 1, 1, padding='VALID', name="avg_pool_736"))

        (self.feed('avg_pool_736')
         .linear(736, 128, relu=False, name='fc_face_vec')
         .dropout(0.5, name='drop1')
         .triplet_pairs_fc(2, name='triplet_fc')
         )

