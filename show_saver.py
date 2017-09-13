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
import mnet_model as model

saver = tf.train.Saver()

model_path = sys.argv[1]

with tf.variable_scope('img_t_net'):
        X = tf.placeholder(tf.float32, shape=img_4d.shape, name='input')
        # X = tf.placeholder(tf.float32, shape=[None,None,None,None], name='input')
        Y = model.net(X, False)

with tf.Session() as sess:
    saver.restore(sess, model_path)
