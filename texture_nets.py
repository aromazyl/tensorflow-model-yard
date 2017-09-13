#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 zhangyule <zyl2336709@gmail.com>
#
# Distributed under terms of the MIT license.

"""

"""
import tensorflow as tf
import numpy as npy

def conv2d(x, input_filters, output_filters, kernel, strides, mode='REFLECT'):
    with tf.variable_scope('conv') as scope:

        shape = [kernel, kernel, input_filters, output_filters]
        weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')
        x_padded = tf.pad(x, [[0, 0], [kernel / 2, kernel / 2], [kernel / 2, kernel / 2], [0, 0]], mode=mode)
        return tf.nn.conv2d(x_padded, weight, strides=[1, strides, strides, 1], padding='VALID', name='conv')


def conv2d_transpose(x, input_filters, output_filters, kernel, strides):
    with tf.variable_scope('conv_transpose') as scope:

        shape = [kernel, kernel, output_filters, input_filters]
        weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')

        batch_size = tf.shape(x)[0]
        height = tf.shape(x)[1] * strides
        width = tf.shape(x)[2] * strides
        output_shape = tf.stack([batch_size, height, width, output_filters])
        return tf.nn.conv2d_transpose(x, weight, output_shape, strides=[1, strides, strides, 1], name='conv_transpose')


def resize(x, name, scale=2):
    with tf.variable_scope('resize-' + name):
        height = x.get_shape()[1].value
        width = x.get_shape()[2].value
        new_height = height * scale
        new_width = width * scale
        return tf.image.resize_images(x, [int(new_height), int(new_width)], tf.image.ResizeMethod.NEAREST_NEIGHBOR)

def resize_conv2d(x, input_filters, output_filters, kernel, strides, training):
    '''
    An alternative to transposed convolution where we first resize, then convolve.
    See http://distill.pub/2016/deconv-checkerboard/

    For some reason the shape needs to be statically known for gradient propagation
    through tf.image.resize_images, but we only know that for fixed image size, so we
    plumb through a "training" argument
    '''
    with tf.variable_scope('conv_transpose') as scope:
        height = x.get_shape()[1].value if training else tf.shape(x)[1]
        width = x.get_shape()[2].value if training else tf.shape(x)[2]

        new_height = height * strides * 2
        new_width = width * strides * 2

        x_resized = tf.image.resize_images(x, [new_height, new_width], tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # shape = [kernel, kernel, input_filters, output_filters]
        # weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')
        return conv2d(x_resized, input_filters, output_filters, kernel, strides)


def instance_norm(x):
    epsilon = 1e-9

    mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)

    return tf.div(tf.subtract(x, mean), tf.sqrt(tf.add(var, epsilon)))


def batch_norm(x, size, training, decay=0.999):
    beta = tf.Variable(tf.zeros([size]), name='beta')
    scale = tf.Variable(tf.ones([size]), name='scale')
    pop_mean = tf.Variable(tf.zeros([size]))
    pop_var = tf.Variable(tf.ones([size]))
    epsilon = 1e-3

    batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
    train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
    train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))

    def batch_statistics():
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, scale, epsilon, name='batch_norm')

    def population_statistics():
        return tf.nn.batch_normalization(x, pop_mean, pop_var, beta, scale, epsilon, name='batch_norm')

    return tf.cond(training, batch_statistics, population_statistics)


def residual(x, filters, kernel, strides):
    with tf.variable_scope('residual') as scope:
        # conv1 = conv2d(x, filters, filters, kernel, strides)
        # conv2 = conv2d(tf.nn.relu(conv1), filters, filters, kernel, strides)
        conv2 = conv2d(tf.nn.relu(x), filters, filters, kernel, strides)

        residual = x + conv2

        return residual

def subnet(image, name, conv_num, ratios, index):
    ratio = ratios[index]
    with tf.variable_scope('conv1-' + name):
        pool = tf.nn.pool(image, (ratio, ratio), 'AVG', padding='SAME')
        pool = resize(pool, name, 1.0 / ratio)
        conv1 = tf.nn.relu(instance_norm(conv2d(pool, 3, conv_num, 3, 1)))

    with tf.variable_scope('conv2-' + name):
        conv2 = tf.nn.relu(instance_norm(conv2d(conv1, conv_num, conv_num, 3, 1)))

    with tf.variable_scope('conv3-' + name):
        conv3 = tf.nn.relu(instance_norm(conv2d(conv2, conv_num, conv_num, 3, 1)))
        if index == 0:
            conv3 = resize(conv3, name)

    return conv3

def connect_net(cur, image, index, ratios, name, conv_num):
    with tf.variable_scope('connect_net-' + name):
        seq0 = subnet(image, 'connect_net-%d' % index, 8, ratios, index)
        seq1 = instance_norm(seq0)
        cur0 = instance_norm(cur)
        cur1 = tf.concat([cur0, seq1], axis=-1)
        def block_helper(img, num, kernel):
            return tf.nn.relu(instance_norm(conv2d(img, conv_num * num, conv_num * num, kernel, 1)))
        cur2 = block_helper(cur1, index + 1, 3)
        cur3 = block_helper(cur2, index + 1, 3)
        cur4 = block_helper(cur3, index + 1, 1)
        if index == len(ratios) - 1:
            cur5 = conv2d(cur4, conv_num * (index+1), 3, 1, 1)
        else:
            cur5 = resize(cur4, name)
    return cur5

def net(image, training=True, ratios=[32,16,8,4,2,1], noise_fn=lambda x:x):
    images = map(noise_fn, [image for i in ratios])
    with tf.variable_scope('texture_net'):
        for index in xrange(len(ratios)):
            if index == 0:
                cur = subnet(images[index], 'subnet-%d'% index, 8, ratios, index)
            else:
                cur = connect_net(cur, images[index], index, ratios, 'texture_layer-' + str(index), 8)
    return cur
