#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:deng
# datetime:2019/7/8 13:05
# software: PyCharm Community Edition

# coding=utf-8

import os
import tensorflow as tf
from PIL import Image
import sys


imgpath = 'D:/Python/data/flowers/17flowers/'
save_path = 'D:/Python/data/flowers/train.tfrecords'

def creat_tf(imgpath, save_path):
    classes = os.listdir(imgpath)

    # 此处定义tfrecords文件存放
    with tf.python_io.TFRecordWriter(save_path) as writer:
        for index, name in enumerate(classes):
            class_path = imgpath + name + "/"
            print(class_path)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    img_path = class_path + img_name
                    img = Image.open(img_path)
                    img = img.resize((224, 224))
                    img_raw = img.tobytes()
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(name)])),
                        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                    }))
                    writer.write(example.SerializeToString())
                    print(img_name)

def read_example():
    # 简单的读取例子：
    for serialized_example in tf.python_io.tf_record_iterator(save_path):
        example = tf.train.Example()
        example.ParseFromString(serialized_example)

        # image = example.features.feature['img_raw'].bytes_list.value
        label = example.features.feature['label'].int64_list.value
        # 可以做一些预处理之类的
        print(label)


if __name__ == '__main__':
    creat_tf(imgpath, save_path)
    # read_example()

