#-*- coding:utf-8 -*-
from utils.point_feature import *
from keras import backend as K
import numpy as np
import tensorflow as tf

def test_session():
    m1 = tf.constant([3, 5])
    m2 = tf.constant([2, 4])
    graph = tf.add(m1, m2)
    graph1 = tf.multiply(graph, m2)

    print(graph)

    with tf.Session() as session:
        with session.as_default():
            print(graph1.eval())
        # print(session.run(graph))


def main():
    test_session()
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
