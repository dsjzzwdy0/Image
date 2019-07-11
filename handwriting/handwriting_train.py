#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:deng
# datetime:2019/7/5 15:35
# software: PyCharm Community Edition

import tensorflow as tf
from handwriting.handwriting import *

xs, ys, keep_prob, train_step, prediction = create_l5_net()

# 定义Session
sess = tf.Session()
init = tf.global_variables_initializer()
# 执行初始化
sess.run(init)

# 进行训练迭代
for i in range(1000):
    # 取出mnist数据集中的100个数据
    batch_xs, batch_ys = mnist.train.next_batch(50)  # 100
    # 执行训练过程并传入真实数据
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 100 == 0:
        print(compute_accuracy(sess, prediction, xs, mnist.test.images, mnist.test.labels, keep_prob))


saver = tf.train.Saver()  # 定义saver
saver.save(sess, model_path)  # 模型储存位置