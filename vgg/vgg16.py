#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:deng
# datetime:2019/7/8 9:56
# software: PyCharm Community Edition

# coding=utf-8

import tensorflow as tf
import numpy as np
from vgg import *


# 加载预训练模型
data_dict = np.load(pre_vgg16_model_path, allow_pickle=True, encoding='latin1').item()


# 打印每层信息
def print_layer(t):
    print(t.op.name, ' ', t.get_shape().as_list())


# 定义卷积层
"""
此处权重初始化定义了3种方式：
    1.预训练模型参数
    2.截尾正态，参考书上采用该方式
    3.xavier，网上blog有采用该方式
通过参数finetrun和xavier控制选择哪种方式，有兴趣的可以都试试
"""


def conv(x, d_out, name, fine_tuning=False, xavier=False):
    d_in = x.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        # Fine-tuning
        if fine_tuning:
            kernel = tf.constant(data_dict[name][0], name="weights")
            bias = tf.constant(data_dict[name][1], name="bias")
            print("finetuning")

        elif not xavier:
            kernel = tf.Variable(tf.truncated_normal([3, 3, d_in, d_out], stddev=0.1), name='weights')
            bias = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[d_out]),
                               trainable=True,
                               name='bias')
            print("truncated_normal")

        else:
            kernel = tf.get_variable(scope + 'weights', shape=[3, 3, d_in, d_out],
                                     dtype=tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer_conv2d())
            bias = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[d_out]),
                               trainable=True,
                               name='bias')
            print("xavier")

        conv = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
        activation = tf.nn.relu(conv + bias, name=scope)
        print_layer(activation)
        return activation


# 最大池化层
def maxpool(x, name):
    activation = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID', name=name)
    print_layer(activation)
    return activation


# 定义全连接层
"""
此处权重初始化定义了3种方式：
    1.预训练模型参数
    2.截尾正态，参考书上采用该方式
    3.xavier，网上blog有采用该方式
通过参数finetrun和xavier控制选择哪种方式，有兴趣的可以都试试
"""


def fc(x, n_out, name, fine_tuning=False, xavier=False):
    n_in = x.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        if fine_tuning:
            weight = tf.constant(data_dict[name][0], name="weights")
            bias = tf.constant(data_dict[name][1], name="bias")
            print("finetuning")

        elif not xavier:
            weight = tf.Variable(tf.truncated_normal([n_in, n_out], stddev=0.01), name='weights')
            bias = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[n_out]),
                               trainable=True,
                               name='bias')
            print("truncated_normal")

        else:
            weight = tf.get_variable(scope + 'weights', shape=[n_in, n_out],
                                     dtype=tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer_conv2d())
            bias = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[n_out]),
                               trainable=True,
                               name='bias')
            print("xavier")

        # 全连接层可以使用relu_layer函数比较方便，不用像卷积层使用relu函数
        activation = tf.nn.relu_layer(x, weight, bias, name=name)
        print_layer(activation)
        return activation


def VGG16(images, _dropout, n_cls):
    """
    此处权重初始化方式采用的是：
        卷积层使用预训练模型中的参数
        全连接层使用xavier类型初始化
    """
    conv1_1 = conv(images, 64, 'conv1_1', fine_tuning=True)
    conv1_2 = conv(conv1_1, 64, 'conv1_2', fine_tuning=True)
    pool1 = maxpool(conv1_2, 'pool1')

    conv2_1 = conv(pool1, 128, 'conv2_1', fine_tuning=True)
    conv2_2 = conv(conv2_1, 128, 'conv2_2', fine_tuning=True)
    pool2 = maxpool(conv2_2, 'pool2')

    conv3_1 = conv(pool2, 256, 'conv3_1', fine_tuning=True)
    conv3_2 = conv(conv3_1, 256, 'conv3_2', fine_tuning=True)
    conv3_3 = conv(conv3_2, 256, 'conv3_3', fine_tuning=True)
    pool3 = maxpool(conv3_3, 'pool3')

    conv4_1 = conv(pool3, 512, 'conv4_1', fine_tuning=True)
    conv4_2 = conv(conv4_1, 512, 'conv4_2', fine_tuning=True)
    conv4_3 = conv(conv4_2, 512, 'conv4_3', fine_tuning=True)
    pool4 = maxpool(conv4_3, 'pool4')

    conv5_1 = conv(pool4, 512, 'conv5_1', fine_tuning=True)
    conv5_2 = conv(conv5_1, 512, 'conv5_2', fine_tuning=True)
    conv5_3 = conv(conv5_2, 512, 'conv5_3', fine_tuning=True)
    pool5 = maxpool(conv5_3, 'pool5')

    '''
    因为训练自己的数据，全连接层最好不要使用预训练参数
    '''
    flatten = tf.reshape(pool5, [-1, 7 * 7 * 512])
    fc6 = fc(flatten, 4096, 'fc6', xavier=True)
    dropout1 = tf.nn.dropout(fc6, _dropout)

    fc7 = fc(dropout1, 4096, 'fc7', xavier=True)
    dropout2 = tf.nn.dropout(fc7, _dropout)

    fc8 = fc(dropout2, n_cls, 'fc8', xavier=True)

    return fc8
