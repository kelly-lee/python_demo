#!/usr/bin/env python
# -*- coding: utf-8 -*-
# coding: utf-8
import sys

# reload(sys)
# sys.setdefaultencoding('utf8')
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], pading='SAME')


def max_pool_2x2(x):
    # x input tensor of shape [batch,in_height,in_width,in_channels]
    # W filter / kernel tensor of shape {filter_height,filter_width,in_channels,out_channels}
    # strides[0]=strides[3] = 1
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

batch_size = 100
n_batch = mnist.train.num_examples
x = tf.placeholder(tf.float32[None, 784])
y = tf.placeholder(tf.floata32[None, 784])
# 改变x的格式为4D向量[batch,in_height,in_width,in_channels]
x_image = tf.reshape(x, [-1, 28, 28, 1])

w1 = weight_variable([5, 5, 1, 32])  # 5*5 的采样窗口，32个卷积核从1个平面抽取特征
b1 = bias_variable([32])  # 每一个卷积核一个偏置值
z1 = conv2d(x_image, w1) + b1
a1 = tf.nn.relu(z1)
pool1 = max_pool_2x2(a1)

w2 = weight_variable([5, 5, 32, 64])  # 5*5 的采样窗口，64个卷积核从32个平面抽取特征
b2 = bias_variable([64])  # 每一个卷积核一个偏置值
z2 = conv2d(pool1, w2) + b2
a2 = tf.nn.relu(z2)
pool2 = max_pool_2x2(a2)

# 28*28的图片第一次卷积后还是28*28，第一次池化后变成14*14
# 第二次卷积后为14*14，第二次池化后变成7*7
# 经过上面操作后得到64张7*7的平面

# 初始化第一个全连接层的权值
w3 = weight_variable([7 * 7 * 64, 1024])
b3 = bias_variable([1024])
pool2_reshape = tf.reshape(pool2, [-1, 7 * 7 * 64])
z3 = tf.matmul(pool2_reshape, w3) + b3
a3 = tf.nn.relu(z3)

keep_prob = tf.placeholder(tf.float32)
dropout3 = tf.nn.dropout(a3, keep_prob)

w4 = weight_variable([1024, 10])
b4 = bias_variable([10])
z4 = tf.matmul(dropout3, w4) + b4
prediction = tf.nn.softmax(z4)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    for epoch in range(21):
        print('epoch', epoch)
        for batch in range(n_batch):
            print('bath', batch)
            batch_xs, batch_ys = mnist.trian.next_batch(batch_size)
            session.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.7})
        acc = session.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
        print ('Iter' + str(epoch) + ',Testing Accuracy=' + str(acc))
