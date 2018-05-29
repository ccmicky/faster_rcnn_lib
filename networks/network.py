#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 ccmicky <chengchun@imdada.cn>
# Licensed under the Dada tech.co.ltd - http://www.imdada.cn

import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
import roi_pooling_layer.roi_pooling_op as roi_pool_op
import roi_pooling_layer.roi_pooling_op_grad
from rpn_msr.proposal_layer_tf import proposal_layer as proposal_layer_py
from rpn_msr.anchor_target_layer_tf import \
    anchor_target_layer as anchor_target_layer_py
from rpn_msr.proposal_target_layer_tf import \
    proposal_target_layer as proposal_target_layer_py
from tensorflow.python.ops import array_ops

DEFAULT_PADDING = 'SAME'


def layer(op):
    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.inputs) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.inputs) == 1:
            layer_input = self.inputs[0]
        else:
            layer_input = list(self.inputs)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self

    return layer_decorated


class Network(object):
    def __init__(self, inputs, trainable=True):
        self.inputs = []
        self.layers = dict(inputs)
        self.trainable = trainable
        self.setup()

    def setup(self):
        raise NotImplementedError('Must be subclassed.')

    def load(self, data_path, session, saver, ignore_missing=False):
        if data_path.endswith('.ckpt'):
            saver.restore(session, data_path)
        else:
            data_dict = np.load(data_path).item()
            for key in data_dict:
                with tf.variable_scope(key, reuse=True):
                    for subkey in data_dict[key]:
                        try:
                            var = tf.get_variable(subkey)
                            session.run(var.assign(data_dict[key][subkey]))
                            print "assign pretrain model " + subkey + " to " + key
                        except ValueError:
                            print "ignore " + key
                            if not ignore_missing:
                                raise

    def feed(self, *args):
        assert len(args) != 0
        self.inputs = []
        for layer in args:
            if isinstance(layer, basestring):
                try:
                    layer = self.layers[layer]
                    print layer
                except KeyError:
                    print self.layers.keys()
                    raise KeyError('Unknown layer name fed: %s' % layer)
            self.inputs.append(layer)
        return self

    def get_output(self, layer):
        try:
            layer = self.layers[layer]
        except KeyError:
            print self.layers.keys()
            raise KeyError('Unknown layer name fed: %s' % layer)
        return layer

    def get_unique_name(self, prefix):
        id = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, id)

    def make_var(self, name, shape, initializer=None, trainable=True):
        return tf.get_variable(name, shape, initializer=initializer,
                               trainable=trainable)

    def validate_padding(self, padding):
        assert padding in ('SAME', 'VALID')

    def batch_norm(self, x, eps=1e-05, decay=0.9, affine=True, name=None,
                   on_train=False):
        """
        on_train = True
        with tf.variable_scope(name) as scope:
            params_shape = x.get_shape()[-1]
            moving_mean = tf.get_variable('mean', params_shape,
                                          initializer=tf.zeros_initializer,
                                          trainable=False)
            moving_variance = tf.get_variable('variance', params_shape,
                                              initializer=tf.ones_initializer,
                                              trainable=False)

            def mean_var_with_update():
                print "abbcbcbcbcbcbcbcbcbcbcbcbcbcbcb"
                mean, variance = tf.nn.moments(x, range(len(x.get_shape())-1),
                                               name='moments')
                with tf.control_dependencies(
                        [assign_moving_average(moving_mean, mean, decay),
                         assign_moving_average(moving_variance, variance,
                                               decay)]):
                    return tf.identity(mean), tf.identity(variance)

            if on_train:

                mean, variance = mean_var_with_update()
            else:
                mean, variance = moving_mean, moving_variance
            if affine:
                beta = tf.get_variable('beta', params_shape,
                                       initializer=tf.zeros_initializer)
                gamma = tf.get_variable('gamma', params_shape,
                                        initializer=tf.ones_initializer)
                x = tf.nn.batch_normalization(x, mean, variance, beta, gamma,
                                              eps)
            else:
                x = tf.nn.batch_normalization(x, mean, variance, None, None,
                                              eps)
        """
        return tf.layers.batch_normalization(x, training=True)

    def conv_relu(self, input, k_h, k_w, c_o, s_h, s_w, name, relu=True,
                  padding='VALID', group=1, trainable=True, batch_norm=True,
                  ontrain=True):
        self.validate_padding(padding)
        c_i = input.get_shape()[-1]
        print "c_i", c_i, input.get_shape()
        assert c_i % group == 0
        assert c_o % group == 0
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1],
                                             padding=padding)
        regularizer = layers.l2_regularizer(0.03)
        with tf.variable_scope(name) as scope:

            init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
            init_biases = tf.constant_initializer(0.0)

            kernel = self.make_var('weights', [k_h, k_w, c_i / group, c_o],
                                   init_weights, trainable)
            biases = self.make_var('biases', [c_o], init_biases, trainable)

            if group == 1:
                conv = convolve(input, kernel)
            else:
                input_groups = tf.split(3, group, input)
                kernel_groups = tf.split(3, group, kernel)
                output_groups = [convolve(i, k) for i, k in
                                 zip(input_groups, kernel_groups)]
                conv = tf.concat(3, output_groups)
            if batch_norm:
                conv = self.batch_norm(conv, name=name + "_bn",
                                       on_train=ontrain)
            if relu:
                bias = tf.nn.bias_add(conv, biases)
                return tf.nn.relu(bias, name=scope.name)
            return tf.nn.bias_add(conv, biases, name=scope.name)

    @layer
    def conv(self, input, k_h, k_w, c_o, s_h, s_w, name, relu=True,
             padding=DEFAULT_PADDING, group=1, trainable=True,
             batch_norm=False, ontrain=True):
        self.validate_padding(padding)
        c_i = input.get_shape()[-1]
        assert c_i % group == 0
        assert c_o % group == 0
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1],
                                             padding=padding)
        regularizer = layers.l2_regularizer(0.03)
        with tf.variable_scope(name) as scope:

            init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
            init_biases = tf.constant_initializer(0.0)
            kernel = self.make_var('weights', [k_h, k_w, c_i / group, c_o],
                                   init_weights, trainable)
            biases = self.make_var('biases', [c_o], init_biases, trainable)

            if group == 1:
                conv = convolve(input, kernel)
            else:
                input_groups = tf.split(3, group, input)
                kernel_groups = tf.split(3, group, kernel)
                output_groups = [convolve(i, k) for i, k in
                                 zip(input_groups, kernel_groups)]
                conv = tf.concat(3, output_groups)
            if batch_norm:
                conv = self.batch_norm(conv, name=name + "_bn",
                                       on_train=ontrain)
            if relu:
                bias = tf.nn.bias_add(conv, biases)
                return tf.nn.relu(bias, name=scope.name)
            return tf.nn.bias_add(conv, biases, name=scope.name)

    @layer
    def relu(self, input, name):
        return tf.nn.relu(input, name=name)

    @layer
    def cross_map_lrn(self, alpha, beta, input, name):
        return tf.nn.lrn(input, alpha=alpha, beta=beta, name=name)

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name,
                 padding=DEFAULT_PADDING):
        self.validate_padding(padding)

        return tf.nn.max_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def avg_pool(self, input, k_h, k_w, s_h, s_w, name,
                 padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.avg_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def roi_pool(self, input, pooled_height, pooled_width, spatial_scale,
                 name):
        # only use the first input
        if isinstance(input[0], tuple):
            input[0] = input[0][0]

        if isinstance(input[1], tuple):
            input[1] = input[1][0]

        print input
        return roi_pool_op.roi_pool(input[0], input[1],
                                    pooled_height,
                                    pooled_width,
                                    spatial_scale,
                                    name=name)[0]

    @layer
    def proposal_layer(self, input, _feat_stride, anchor_scales, cfg_key,
                       name):
        if isinstance(input[0], tuple):
            input[0] = input[0][0]
        return tf.reshape(tf.py_func(proposal_layer_py,
                                     [input[0], input[1], input[2], cfg_key,
                                      _feat_stride, anchor_scales],
                                     [tf.float32]), [-1, 5], name=name)

    @layer
    def anchor_target_layer(self, input, _feat_stride, anchor_scales, name):
        if isinstance(input[0], tuple):
            input[0] = input[0][0]

        with tf.variable_scope(name) as scope:
            rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = tf.py_func(
                anchor_target_layer_py,
                [input[0], input[1], input[2], input[3], _feat_stride,
                 anchor_scales],
                [tf.float32, tf.float32, tf.float32, tf.float32])

            rpn_labels = tf.convert_to_tensor(tf.cast(rpn_labels, tf.int32),
                                              name='rpn_labels')
            rpn_bbox_targets = tf.convert_to_tensor(rpn_bbox_targets,
                                                    name='rpn_bbox_targets')
            rpn_bbox_inside_weights = tf.convert_to_tensor(
                rpn_bbox_inside_weights, name='rpn_bbox_inside_weights')
            rpn_bbox_outside_weights = tf.convert_to_tensor(
                rpn_bbox_outside_weights, name='rpn_bbox_outside_weights')

            return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights

    @layer
    def proposal_target_layer(self, input, classes, name):
        if isinstance(input[0], tuple):
            input[0] = input[0][0]
        with tf.variable_scope(name) as scope:
            rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = tf.py_func(
                proposal_target_layer_py, [input[0], input[1], classes],
                [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])

            rois = tf.reshape(rois, [-1, 5], name='rois')
            labels = tf.convert_to_tensor(tf.cast(labels, tf.int32),
                                          name='labels')
            bbox_targets = tf.convert_to_tensor(bbox_targets,
                                                name='bbox_targets')
            bbox_inside_weights = tf.convert_to_tensor(bbox_inside_weights,
                                                       name='bbox_inside_weights')
            bbox_outside_weights = tf.convert_to_tensor(bbox_outside_weights,
                                                        name='bbox_outside_weights')

            return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights

    @layer
    def reshape_layer(self, input, d, name):
        input_shape = tf.shape(input)
        if name == 'rpn_cls_prob_reshape':
            return tf.transpose(
                tf.reshape(tf.transpose(input, [0, 3, 1, 2]), [input_shape[0],
                                                               int(d), tf.cast(
                        tf.cast(input_shape[1], tf.float32) / tf.cast(d,
                                                                      tf.float32) * tf.cast(
                            input_shape[3], tf.float32), tf.int32),
                                                               input_shape[
                                                                   2]]),
                [0, 2, 3, 1], name=name)
        else:
            return tf.transpose(
                tf.reshape(tf.transpose(input, [0, 3, 1, 2]), [input_shape[0],
                                                               int(d), tf.cast(
                        tf.cast(input_shape[1], tf.float32) * (
                            tf.cast(input_shape[3], tf.float32) / tf.cast(d,
                                                                          tf.float32)),
                        tf.int32), input_shape[2]]), [0, 2, 3, 1], name=name)

    @layer
    def feature_extrapolating(self, input, scales_base, num_scale_base,
                              num_per_octave, name):
        return feature_extrapolating_op.feature_extrapolating(input,
                                                              scales_base,
                                                              num_scale_base,
                                                              num_per_octave,
                                                              name=name)

    @layer
    def lrn(self, input, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(input,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name=name)

    @layer
    def concat(self, inputs, axis, name):
        return tf.concat(concat_dim=axis, values=inputs, name=name)

    @layer
    def linear(self, input, num_in, num_out, name, relu=True, trainable=True):
        regularizer = layers.l2_regularizer(0.03)
        with tf.variable_scope(name, regularizer=regularizer) as scope:
            # only use the first input
            if isinstance(input, tuple):
                input = input[0]

            input_shape = input.get_shape()
            print "aaaaaaaaaaaaaaaaaaaaaaa", "input_shape", input_shape
            if input_shape.ndims == 4:
                dim = num_in
                feed_in = tf.reshape(tf.transpose(input, [0, 3, 1, 2]),
                                     [-1, dim])
            else:
                feed_in, dim = (input, int(input_shape[-1]))

            init_weights = tf.truncated_normal_initializer(0.0,
                                                           stddev=0.01)
            init_biases = tf.constant_initializer(0.0)

            weights = self.make_var('weights', [dim, num_out], init_weights,
                                    trainable)
            biases = self.make_var('biases', [num_out], init_biases, trainable)

            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=scope.name)
            return fc

    @layer
    def fc(self, input, num_out, name, relu=True, trainable=True):
        with tf.variable_scope(name) as scope:
            # only use the first input
            if isinstance(input, tuple):
                input = input[0]

            input_shape = input.get_shape()
            print "aaaaaaaaaaaaaaaaaaaaaaa", "input_shape", input_shape
            if input_shape.ndims == 4:
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(tf.transpose(input, [0, 3, 1, 2]),
                                     [-1, dim])
            else:
                feed_in, dim = (input, int(input_shape[-1]))

            if name == 'bbox_pred':
                init_weights = tf.truncated_normal_initializer(0.0,
                                                               stddev=0.001)
                init_biases = tf.constant_initializer(0.0)
            else:
                init_weights = tf.truncated_normal_initializer(0.0,
                                                               stddev=0.01)
                init_biases = tf.constant_initializer(0.0)

            weights = self.make_var('weights', [dim, num_out], init_weights,
                                    trainable)
            biases = self.make_var('biases', [num_out], init_biases, trainable)

            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=scope.name)
            return fc

    @layer
    def softmax(self, input, name):
        input_shape = tf.shape(input)
        if name == 'rpn_cls_prob':
            return tf.reshape(
                tf.nn.softmax(tf.reshape(input, [-1, input_shape[3]])),
                [-1, input_shape[1], input_shape[2], input_shape[3]],
                name=name)
        else:
            return tf.nn.softmax(input, name=name)

    @layer
    def triplet_pairs_fc(self, input, num_out, name, relu=True,
                         trainable=True):
        rows = tf.cast(tf.divide(tf.cast(tf.shape(input)[0], tf.int32), 3),
                       tf.int32)
        mat_a = input[:rows, :]
        mat_p = input[rows:2 * rows, :]
        mat_n = input[2 * rows:, :]
        mat_a_p = tf.concat([mat_a, mat_p], 1)
        mat_a_n = tf.concat([mat_a, mat_n], 1)
        mat = tf.concat([mat_a_p, mat_a_n], 0)
        dim = int(mat.get_shape()[-1])
        regularizer = layers.l2_regularizer(0.03)
        with tf.variable_scope(name, regularizer=regularizer) as scope:
            init_weights = tf.truncated_normal_initializer(0.0,
                                                           stddev=0.01)
            init_biases = tf.constant_initializer(0.0)
            weights = self.make_var('weights', [dim, num_out], init_weights,
                                    trainable)
            biases = self.make_var('biases', [num_out], init_biases, trainable)

            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(mat, weights, biases)
            return fc

    @layer
    def dropout(self, input, keep_prob, name):
        return tf.nn.dropout(input, keep_prob, name=name)

    @layer
    def Inception(self, input, kernelSize, kernelStride, outputSize,
                  reduceSize, pool_k_h, pool_k_w, pool_s_h, pool_s_w, name,
                  ontrain=True):
        x = []
        depth_dim = 0
        input_shape = input.get_shape()
        target_size = [0] * len(input_shape.as_list())

        # with tf.variable_scope(name) as scope:
        for i in range(len(kernelSize)):
            x_1 = self.conv_relu(input, 1, 1, reduceSize[i], 1, 1,
                                 name + '_' + str(i) + '_0', batch_norm=True,
                                 ontrain=ontrain)
            x_1 = self.conv_relu(x_1, kernelSize[i], kernelSize[i],
                                 outputSize[i], kernelStride[i],
                                 kernelStride[i],
                                 name + '_' + str(i) + '_1', batch_norm=True,
                                 ontrain=ontrain)
            x.append(x_1)

        x_2 = tf.nn.max_pool(input,
                             ksize=[1, pool_k_h, pool_k_w, 1],
                             strides=[1, pool_s_h, pool_s_w, 1],
                             padding='SAME')
        ii = len(kernelSize)
        if ii < len(reduceSize) and reduceSize[ii] is not None:
            i = ii
            x_2 = self.conv_relu(x_2, 1, 1, reduceSize[i], 1, 1,
                                 name + '_' + str(i) + '_2', batch_norm=True,
                                 ontrain=ontrain)
        x.append(x_2)

        ii += 1
        if ii < len(reduceSize) and reduceSize[ii] is not None:
            i = ii
            x_3 = self.conv_relu(input, 1, 1, reduceSize[i], 1, 1,
                                 name + '_' + str(i) + '_3', batch_norm=True,
                                 ontrain=ontrain)
            x.append(x_3)

        for i in range(len(x)):
            print i, x[i].shape[-1].value, x
            x_size = tf.shape(x[i])
            for j in range(len(target_size)):
                target_size[j] = tf.py_func(np.maximum,
                                            [target_size[j],
                                             tf.cast(x_size[j], tf.int32)],
                                            tf.int32)
            depth_dim += x[i].shape[-1].value

        for i in range(len(x)):
            paddings = [[0, 0], [0, 0], [0, 0], [0, 0]]
            x_size = tf.shape(x[i])
            for j in range(1, 3):
                pad_1 = tf.py_func(np.divide, [
                    tf.py_func(np.subtract,
                               [target_size[j],
                                tf.cast(x_size[j], tf.int32)],
                               tf.int32), 2], tf.int32)
                pad_2 = tf.py_func(np.subtract, [
                    tf.py_func(np.subtract,
                               [target_size[j],
                                tf.cast(x_size[j], tf.int32)],
                               tf.int32), pad_1], tf.int32)
                paddings[j] = [pad_1, pad_2]
            # paddings = tf.convert_to_tensor(paddings)
            x[i] = tf.pad(x[i], paddings, "CONSTANT")
        x = tf.concat(x, 3)
        x_shape = tf.shape(x)
        return tf.transpose(tf.reshape(tf.transpose(x, [0, 3, 1, 2]),
                                       [x_shape[0], int(depth_dim), tf.cast(
                                           tf.cast(x_shape[1], tf.float32) * (
                                               tf.cast(x_shape[3],
                                                       tf.float32) / tf.cast(
                                                   depth_dim, tf.float32)),
                                           tf.int32), x_shape[2]]),
                            [0, 2, 3, 1], name=name)


