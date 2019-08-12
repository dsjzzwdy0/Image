#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/7/7 15:00
# @Author : Deng Shijun
# @Site : 
# @File : tf_test.py
# @Software: PyCharm
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

#case 1
# input_1 = tf.Variable(tf.random_normal([1, 3, 3, 5]))
'''
# case 2
input = tf.Variable(tf.random_normal([1, 3, 3, 5]))
filter = tf.Variable(tf.random_normal([1, 1, 5, 1]))

op2 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')
'''
'''
# case 3
input = tf.Variable(tf.random_normal([1, 3, 3, 5]))
filter = tf.Variable(tf.random_normal([3, 3, 5, 1]))

op3 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')
# case 4
input = tf.Variable(tf.random_normal([1, 5, 5, 5]))
filter = tf.Variable(tf.random_normal([3, 3, 5, 1]))

op4 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')
# case 5
input = tf.Variable(tf.random_normal([1, 5, 5, 5]))
filter = tf.Variable(tf.random_normal([3, 3, 5, 1]))

op5 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')
# case 6
input = tf.Variable(tf.random_normal([1, 5, 5, 5]))
filter = tf.Variable(tf.random_normal([3, 3, 5, 7]))

op6 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')
# case 7
input = tf.Variable(tf.random_normal([1, 5, 5, 5]))
filter = tf.Variable(tf.random_normal([3, 3, 5, 7]))

op7 = tf.nn.conv2d(input, filter, strides=[1, 2, 2, 1], padding='SAME')
'''

# case 8
input = tf.Variable(tf.random_normal([10, 5, 5, 5]))
filter = tf.Variable(tf.random_normal([3, 3, 5, 7]))

op8 = tf.nn.conv2d(input, filter, strides=[1, 2, 2, 1], padding='SAME')


init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print("Input Value")
    i = sess.run(input)
    print(i)
    print(i.shape)

    print("Filter Value")
    f = sess.run(filter)
    print(f)
    print(f.shape)

    print("Operator")
    result = sess.run(op8)
    print(result)
    print(result.shape)

    '''print("case 3")
    print(sess.run(op3))
    print("case 4")
    print(sess.run(op4))
    print("case 5")
    print(sess.run(op5))
    print("case 6")
    print(sess.run(op6))
    print("case 7")
    print(sess.run(op7))
    print("case 8")
    print(sess.run(op8))
    '''
