#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:deng
# datetime:2019/7/5 15:35
# software: PyCharm Community Edition
import time
import tensorflow as tf
from cnn.handwriting import *


start = time.time()
xs, ys, keep_prob, train_step, predict = create_l5_net()
end = time.time()
print("Create CNN spend time is", (end - start), "s.")

# 定义Session
sess = tf.Session()
init = tf.global_variables_initializer()
# 执行初始化
sess.run(init)

saver = tf.train.Saver()
# accuracy = tf.reduce_mean(tf.cast(predict, "float"))


def get_label_name(label):
    size = len(label)
    max_prob = -1
    index = -1
    for i in range(size):
        if max_prob < label[i]:
            max_prob = label[i]
            index = i

    return index, max_prob


with tf.Session() as sess:
    start = time.time()
    saver.restore(sess, model_path)
    end = time.time()
    print("Restore CNN spend time is", (end - start), "s.")

    images = mnist.test.images
    labels = mnist.test.labels

    images = images[:100]
    labels = labels[:100]

    size = len(labels)
    start = time.time()
    y_predict = sess.run(predict, feed_dict={xs: images, keep_prob: 1})        ### 做出预测传入要预测的图片xxxxx
    end = time.time()
    print("Predict ", size, " objects spend time is", (end - start), "s.")

    size = len(labels)
    for i in range(size):
        print('Source is:', get_label_name(labels[i]), ", predict is: ", get_label_name(y_predict[i]))

    '''
    start = time.time()
    print(compute_accuracy(sess, predict, xs, mnist.test.images, mnist.test.labels, keep_prob))
    end = time.time()
    print("Compute ", size, " objects predict accuracy spend time is", (end - start), "s.")
    '''

    '''print('test accuracy %g' % accuracy.eval(feed_dict={                ###打印训练好的模型和测试集相比的准确率
        xs: mnist.test.images, ys: mnist.test.labels, keep_prob: 1.0}))'''
