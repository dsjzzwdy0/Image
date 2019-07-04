#-*- coding:utf-8 -*-
from utils.point_feature import *
from keras import backend as K
import numpy as np
import tensorflow as tf
import cv2 as cv


def test_tensorflow():
    t = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]], [[5, 5, 5], [6, 6, 6]]])
    k = tf.slice(t, [1, 0, 0], [1, 2, -1])

    src = cv.imread("D:/python/images/test0.jpg")
    print("Source images shape is", src.shape)
    cv.imshow("input", src)
    cv.waitKey(0)

    _image = tf.placeholder(shape=[None, None, 3], dtype=tf.uint8, name="image")
    roi_image = tf.slice(_image, [40, 130, 0], [580, 280, -1])

    with tf.Session() as sess:
        slice = sess.run(roi_image, feed_dict={_image:src})
        print(slice.shape)
        cv.imshow("roi", slice)
        cv.waitKey(0)
        cv.destroyAllWindows()

        print(sess.run(k))


def test_session():
    '''
    m1 = tf.constant([3, 5])
    m2 = tf.constant([2, 4])
    graph = tf.add(m1, m2)
    graph1 = tf.multiply(graph, m2)
    print(graph)
    '''

    a = tf.placeholder(tf.float32, shape=[2], name=None)
    b = tf.placeholder(tf.float32, shape=[2], name=None)
    c = tf.add(a, b)


    with tf.Session() as session:
        print(session.run(c, feed_dict={a: [10, 20], b:[100, 20]}))
        # with session.as_default():
        #     print(graph1.eval())
        # print(session.run(graph))


def main():
    test_tensorflow()
    # test_session()
    '''b = K.random_uniform_variable(shape=(3, 4), low=0, high=1) # 均匀分布
    c = K.random_normal_variable(shape=(3, 4), mean=0, scale=1) # 高斯分布

    print(b)
    print(c)

    inputs = K.placeholder(shape=(2, 4, 5))
    print(inputs)

    # 全 0 变量：
    var = K.ones(shape=(3, 4, 5))
    print(var)
    print(var.shape)
    print(K.eval(var))

    kvar = K.variable(np.array([[1, 2], [3, 4]]), dtype='float32')
    print(kvar)
    print(kvar.shape)
    print(kvar.value())

    # print("Test for the image.")
    # load_frozen_model('data/geodesc.pb', print_nodes=True)
    val = np.random.random((3, 4, 5))
    print(val.shape)
    print(val)'''


if __name__ == '__main__':
    main()
