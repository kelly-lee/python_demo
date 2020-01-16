#!/usr/bin/env python
# -*- coding: utf-8 -*-
# coding: utf-8
import sys
#
# reload(sys)
# sys.setdefaultencoding('utf8')
import os

from tensorflow.contrib.tensorboard.plugins import projector

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt


def f1():
    # 创建常量op
    m1 = tf.constant([[3, 3]])
    # 创建常量op
    m2 = tf.constant([[2], [3]])
    # 创建矩阵乘法op
    product = tf.matmul(m1, m2)
    print(product)

    # 定义一个会话，启动默认图
    session = tf.Session()
    # 调用session的run方法来执行矩阵乘法op
    # run触发图中三个op
    result = session.run(product)
    print(result)
    session.close()

    with tf.Session() as session:
        # 调用session的run方法来执行矩阵乘法op
        # run触发图中三个op
        result = session.run(product)
        print(result)


# -------------------------------------------------
def f2():
    # 定义变量
    x = tf.Variable([1, 2])
    a = tf.constant([3, 3])
    # 增加减法op
    sub = tf.subtract(x, a)
    # 增加加法op
    add = tf.add(x, sub)
    # 初始化变量
    init = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init)
        print(session.run(sub))
        print(session.run(add))


# -------------------------------------------------
def f3():
    # 创建一个变量初始化为0
    state = tf.Variable(0, name='counter')
    new_value = tf.add(state, 1)
    # 赋值op
    update = tf.assign(state, new_value)
    # 变量初始化
    init = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init)
        print(session.run(state))
        for _ in range(5):
            session.run(update)
            print(session.run(state))


# -------------------------------------------------
def f4():
    # Fetch
    input1 = tf.constant(3.0)
    input2 = tf.constant(2.0)
    input3 = tf.constant(5.0)
    add = tf.add(input2, input3)
    mul = tf.multiply(input1, add)
    with tf.Session() as session:
        result = session.run([mul, add])
        print(result)


# -------------------------------------------------
def f5():
    # Feed
    # 创建占位符
    input1 = tf.placeholder(tf.float32)
    input2 = tf.placeholder(tf.float32)
    output = tf.multiply(input1, input2)

    with tf.Session() as session:
        print(session.run(output, feed_dict={input1: [7, ], input2: [2, ]}))


# -------------------------------------------------
# 线性回归
def f6():
    x_data = np.random.rand(100)
    y_data = x_data * 0.1 + 0.2

    b = tf.Variable(0.)
    k = tf.Variable(0.)
    y = k * x_data + b
    # 求平均值
    loss = tf.reduce_mean(tf.square(y_data - y))
    optimizer = tf.train.GradientDescentOptimizer(0.2)
    train = optimizer.minimize(loss)
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)
        for step in range(201):
            session.run(train)
            if step % 20 == 0:
                print(step, session.run([k, b]))


# -------------------------------------------------
# 非线性回归
def f7():
    x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
    noise = np.random.normal(0, 0.02, x_data.shape)
    y_data = np.square(x_data) + noise

    x = tf.placeholder(tf.float32, [None, 1])
    y = tf.placeholder(tf.float32, [None, 1])

    w1 = tf.Variable(tf.random_normal([1, 10]))
    b1 = tf.Variable(tf.zeros([1, 10]))
    z1 = tf.matmul(x, w1) + b1
    a1 = tf.nn.tanh(z1)

    w2 = tf.Variable(tf.random_normal([10, 1]))
    # 为什么维度是 1,1
    b2 = tf.Variable(tf.zeros([1, 1]))
    z2 = tf.matmul(a1, w2) + b2
    prediction = tf.nn.tanh(z2)

    loss = tf.reduce_mean(tf.square(y - prediction))
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    init = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init)
        for epoch in range(2000):
            session.run(train_step, feed_dict={x: x_data, y: y_data})
        prediction_value = session.run(prediction, feed_dict={x: x_data})
        plt.figure()
        plt.scatter(x_data, y_data)
        plt.plot(x_data, prediction_value, 'r-', lw=5)
        plt.show()


# -----------------------------------------------------
# 图像识别
def f8():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    batch_size = 100
    n_batch = mnist.train.num_examples // batch_size
    x = tf.placeholder(tf.float32, [None, 28 * 28])
    y = tf.placeholder(tf.float32, [None, 10])

    W = tf.Variable(tf.truncated_normal([28 * 28, 10], stddev=0.1))
    b = tf.Variable(tf.zeros([10]) + 0.1)
    prediction = tf.nn.softmax(tf.matmul(x, W) + b)

    # loss = tf.reduce_mean(tf.square(y - prediction))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

    train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
    init = tf.global_variables_initializer()
    # 计算结果存放到一个布尔型列表中
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))  # argmax返回一维张量中最大的值所在的位置
    # 求准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    with tf.Session() as session:
        session.run(init)
        for epoch in range(21):
            for batch in range(n_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                session.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
            acc = session.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
            print("iter" + str(epoch) + ",Testing Accurary " + str(acc))


# 图像识别 dropout
def f9():  # 0.9587
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    batch_size = 100
    n_batch = mnist.train.num_examples // batch_size
    x = tf.placeholder(tf.float32, [None, 28 * 28])
    y = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)

    W1 = tf.Variable(tf.truncated_normal([28 * 28, 200], stddev=0.1))
    b1 = tf.Variable(tf.zeros([200]) + 0.1)
    z1 = tf.matmul(x, W1) + b1
    a1 = tf.nn.tanh(z1)
    dropout1 = tf.nn.dropout(a1, keep_prob)

    W2 = tf.Variable(tf.truncated_normal([200, 200], stddev=0.1))
    b2 = tf.Variable(tf.zeros([200]) + 0.1)
    z2 = tf.matmul(dropout1, W2) + b2
    a2 = tf.nn.tanh(z2)
    dropout2 = tf.nn.dropout(a2, keep_prob)

    W3 = tf.Variable(tf.truncated_normal([200, 100], stddev=0.1))
    b3 = tf.Variable(tf.zeros([100]) + 0.1)
    z3 = tf.matmul(dropout2, W3) + b3
    a3 = tf.nn.tanh(z3)
    dropout3 = tf.nn.dropout(a3, keep_prob)

    W4 = tf.Variable(tf.truncated_normal([100, 10], stddev=0.1))
    b4 = tf.Variable(tf.zeros([10]) + 0.1)
    prediction = tf.nn.softmax(tf.matmul(dropout3, W4) + b4)

    # loss = tf.reduce_mean(tf.square(y - prediction))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

    train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
    init = tf.global_variables_initializer()
    # 计算结果存放到一个布尔型列表中
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))  # argmax返回一维张量中最大的值所在的位置
    # 求准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    with tf.Session() as session:
        session.run(init)
        for epoch in range(21):
            for batch in range(n_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                # session.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
                session.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.7})
            test_acc = session.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
            train_acc = session.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels, keep_prob: 1.0})
            print("iter" + str(epoch) + ",Testing Accurary " + str(test_acc) + ",Training Accurary " + str(train_acc))


# 图像识别 optimizer
def f10():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    batch_size = 100
    n_batch = mnist.train.num_examples // batch_size
    x = tf.placeholder(tf.float32, [None, 28 * 28])
    y = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)

    W1 = tf.Variable(tf.truncated_normal([28 * 28, 500], stddev=0.1))
    b1 = tf.Variable(tf.zeros([500]) + 0.1)
    z1 = tf.matmul(x, W1) + b1
    a1 = tf.nn.tanh(z1)
    dropout1 = tf.nn.dropout(a1, keep_prob)

    W2 = tf.Variable(tf.truncated_normal([500, 300], stddev=0.1))
    b2 = tf.Variable(tf.zeros([300]) + 0.1)
    z2 = tf.matmul(dropout1, W2) + b2
    a2 = tf.nn.tanh(z2)
    dropout2 = tf.nn.dropout(a2, keep_prob)

    W3 = tf.Variable(tf.truncated_normal([300, 10], stddev=0.1))
    b3 = tf.Variable(tf.zeros([10]) + 0.1)
    prediction = tf.nn.softmax(tf.matmul(dropout2, W3) + b3)

    # loss = tf.reduce_mean(tf.square(y - prediction))
    # 交叉熵加快收敛速度
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

    # tf.train.GradientDescentOptimizer
    # tf.train.AdadeltaOptimizer iter20,Testing Accurary 0.834,Training Accurary 0.82394546
    # tf.train.AdagradOptimizer
    # tf.train.AdagradDAOptimizer
    # tf.train.MomentumOptimizer
    # tf.train.AdamOptimizer iter20,Testing Accurary 0.8973,Training Accurary 0.89230907
    # tf.train.FtrlOptimizer
    # tf.train.ProximalGradientDescentOptimizer
    # tf.train.ProximalAdagradOptimizer
    # tf.train.RMSPropOptimizer
    init = tf.global_variables_initializer()
    # 计算结果存放到一个布尔型列表中
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))  # argmax返回一维张量中最大的值所在的位置
    # 求准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    with tf.Session() as session:
        session.run(init)
        for epoch in range(51):
            for batch in range(n_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                # session.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
                session.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.7})
            test_acc = session.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
            train_acc = session.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels, keep_prob: 1.0})
            print("iter" + str(epoch) + ",Testing Accurary " + str(test_acc) + ",Training Accurary " + str(train_acc))


# 图像识别 动态leanring rate
def f11():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    batch_size = 100
    n_batch = mnist.train.num_examples // batch_size
    x = tf.placeholder(tf.float32, [None, 28 * 28])
    y = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)

    lr = tf.Variable(initial_value=0.001, dtype=tf.float32)

    W1 = tf.Variable(tf.truncated_normal([28 * 28, 500], stddev=0.1))
    b1 = tf.Variable(tf.zeros([500]) + 0.1)
    z1 = tf.matmul(x, W1) + b1
    a1 = tf.nn.tanh(z1)
    dropout1 = tf.nn.dropout(a1, keep_prob)

    W2 = tf.Variable(tf.truncated_normal([500, 300], stddev=0.1))
    b2 = tf.Variable(tf.zeros([300]) + 0.1)
    z2 = tf.matmul(dropout1, W2) + b2
    a2 = tf.nn.tanh(z2)
    dropout2 = tf.nn.dropout(a2, keep_prob)

    W3 = tf.Variable(tf.truncated_normal([300, 10], stddev=0.1))
    b3 = tf.Variable(tf.zeros([10]) + 0.1)
    prediction = tf.nn.softmax(tf.matmul(dropout2, W3) + b3)

    # loss = tf.reduce_mean(tf.square(y - prediction))
    # 交叉熵加快收敛速度
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

    train_step = tf.train.AdamOptimizer(lr).minimize(loss)

    init = tf.global_variables_initializer()
    # 计算结果存放到一个布尔型列表中
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))  # argmax返回一维张量中最大的值所在的位置
    # 求准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    with tf.Session() as session:
        session.run(init)
        for epoch in range(21):
            session.run(tf.assign(lr, 0.001 * (0.95 ** epoch)))
            for batch in range(n_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                # session.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
                session.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.7})
            test_acc = session.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
            train_acc = session.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels, keep_prob: 1.0})
            learning_rate = session.run(lr)
            print("iter" + str(epoch) + ",Testing Accurary " + str(test_acc) + ",Training Accurary " + str(
                train_acc) + "， Learning Rate=" + str(learning_rate))


# tensorboard
# console：tensorboard --logdir='/Users/a1800101471/PycharmProjects/python_demo2/tf_demo'
def f12():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    batch_size = 100
    n_batch = mnist.train.num_examples // batch_size

    with tf.name_scope('input_layer'):
        x = tf.placeholder(tf.float32, [None, 28 * 28], name='x-input')
        y = tf.placeholder(tf.float32, [None, 10], name='y-input')
        keep_prob = tf.placeholder(tf.float32)

    with tf.name_scope('hidden_layer'):
        with tf.name_scope('hidden_layer_1'):
            W1 = tf.Variable(tf.truncated_normal([28 * 28, 500], stddev=0.1), name='w1')
            variable_summaries(W1)
            b1 = tf.Variable(tf.zeros([500]) + 0.1, name='b1')
            variable_summaries(b1)
            z1 = tf.add(tf.matmul(x, W1), b1)
            a1 = tf.nn.tanh(z1)
            dropout1 = tf.nn.dropout(a1, keep_prob)

        with tf.name_scope('hidden_layer_2'):
            W2 = tf.Variable(tf.truncated_normal([500, 300], stddev=0.1), name='w2')
            b2 = tf.Variable(tf.zeros([300]) + 0.1, name='b2')
            z2 = tf.matmul(dropout1, W2) + b2
            a2 = tf.nn.tanh(z2)
            dropout2 = tf.nn.dropout(a2, keep_prob)

    with tf.name_scope('output_layer'):
        W3 = tf.Variable(tf.truncated_normal([300, 10], stddev=0.1), name='w3')
        b3 = tf.Variable(tf.zeros([10]) + 0.1, name='b3')
        z3 = tf.matmul(dropout2, W3) + b3
        prediction = tf.nn.softmax(z3)

    # with tf.name_scope('loss'):
    with tf.name_scope('train'):
        lr = tf.Variable(initial_value=0.001, dtype=tf.float32)
        # loss = tf.reduce_mean(tf.square(y - prediction))
        # 交叉熵加快收敛速度
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
        tf.summary.scalar('loss', loss)
        train_step = tf.train.AdamOptimizer(lr).minimize(loss)

    with tf.name_scope('metric'):
        # 计算结果存放到一个布尔型列表中
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))  # argmax返回一维张量中最大的值所在的位置
        # 求准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    with tf.name_scope('init'):
        init = tf.global_variables_initializer()

    # 合并检测
    merged = tf.summary.merge_all()

    with tf.Session() as session:
        session.run(init)
        writer = tf.summary.FileWriter('logs/', session.graph)
        for epoch in range(10):
            session.run(tf.assign(lr, 0.001 * (0.95 ** epoch)))
            for batch in range(n_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                # session.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})

                summary, _ = session.run([merged, train_step], feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.7})
            writer.add_summary(summary, epoch)
            test_acc = session.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
            train_acc = session.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels, keep_prob: 1.0})
            learning_rate = session.run(lr)
            print("iter" + str(epoch) + ",Testing Accurary " + str(test_acc) + ",Training Accurary " + str(
                train_acc) + "， Learning Rate=" + str(learning_rate))


# tensorboard embedding visualisation
# console：tensorboard --logdir='/Users/a1800101471/PycharmProjects/python_demo2/tf_demo'
def f13():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    print('aa')
    # 运行次数
    max_steps = 1001
    image_num = 3000  # 最多不超过1w
    batch_size = 100
    DIR = '/Users/a1800101471/PycharmProjects/python_demo2/tf_demo/'
    session = tf.Session()

    embedding = tf.Variable(tf.stack(mnist.test.images[:image_num]), trainable=False, name='embedding')

    # batch_size = 100
    # n_batch = mnist.train.num_examples // batch_size

    with tf.name_scope('input_layer'):
        x = tf.placeholder(tf.float32, [None, 28 * 28], name='x-input')
        y = tf.placeholder(tf.float32, [None, 10], name='y-input')
        keep_prob = tf.placeholder(tf.float32)

    # 显示图片
    with tf.name_scope('input_reshape'):
        # 黑白图片是1，彩色图片是3
        image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('input', image_shaped_input, 10)

    with tf.name_scope('hidden_layer'):
        with tf.name_scope('hidden_layer_1'):
            W1 = tf.Variable(tf.truncated_normal([28 * 28, 500], stddev=0.1), name='w1')
            variable_summaries(W1)
            b1 = tf.Variable(tf.zeros([500]) + 0.1, name='b1')
            variable_summaries(b1)
            z1 = tf.add(tf.matmul(x, W1), b1)
            a1 = tf.nn.tanh(z1)
            dropout1 = tf.nn.dropout(a1, keep_prob)

        with tf.name_scope('hidden_layer_2'):
            W2 = tf.Variable(tf.truncated_normal([500, 300], stddev=0.1), name='w2')
            b2 = tf.Variable(tf.zeros([300]) + 0.1, name='b2')
            z2 = tf.matmul(dropout1, W2) + b2
            a2 = tf.nn.tanh(z2)
            dropout2 = tf.nn.dropout(a2, keep_prob)

    with tf.name_scope('output_layer'):
        W3 = tf.Variable(tf.truncated_normal([300, 10], stddev=0.1), name='w3')
        b3 = tf.Variable(tf.zeros([10]) + 0.1, name='b3')
        z3 = tf.matmul(dropout2, W3) + b3
        prediction = tf.nn.softmax(z3)

    # with tf.name_scope('loss'):
    with tf.name_scope('train'):
        lr = tf.Variable(initial_value=0.001, dtype=tf.float32)
        # loss = tf.reduce_mean(tf.square(y - prediction))
        # 交叉熵加快收敛速度
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
        tf.summary.scalar('loss', loss)
        train_step = tf.train.AdamOptimizer(lr).minimize(loss)

    with tf.name_scope('metric'):
        # 计算结果存放到一个布尔型列表中
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))  # argmax返回一维张量中最大的值所在的位置
        # 求准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    with tf.name_scope('init'):
        init = tf.global_variables_initializer()
    session.run(init)

    # 产生metadata
    if tf.gfile.Exists(DIR + 'projector/projector/metadata.tsv'):
        tf.gfile.DeleteRecursively(DIR + 'projector/projector/metadata.tsv')
    with open(DIR + 'projector/projector/metadata.tsv', 'w') as f:
        labels = session.run(tf.argmax(mnist.test.labels[:], 1))
        for i in range(image_num):
            f.write(str(labels[i]) + '\n')

    # 合并检测
    merged = tf.summary.merge_all()

    projector_writer = tf.summary.FileWriter(DIR + 'projector/projector', session.graph)
    # 保存网络模型
    saver = tf.train.Saver()
    config = projector.ProjectorConfig()
    embed = config.embeddings.add()
    embed.tensor_name = embedding.name
    embed.metadata_path = DIR + 'projector/projector/metadata.tsv'
    embed.sprite.image_path = DIR + 'projector/data/mnist_10k_sprite.png'
    embed.sprite.single_image_dim.extend([28, 28])
    projector.visualize_embeddings(projector_writer, config)

    for i in range(max_steps):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, _ = session.run([merged, train_step],
                                 feed_dict={x: batch_xs, y: batch_ys,
                                            keep_prob: 0.7}, options=run_options, run_metadata=run_metadata)
        projector_writer.add_run_metadata(run_metadata, 'step%03d' % i)
        projector_writer.add_summary(summary, i)
        if i % 100 == 0:
            acc = session.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels,
                                                   keep_prob: 1.0})
            print('Iter' + str(i) + ',Testing Acuracy=' + str(acc))
    saver.save(session, DIR + 'projector/projector/a_model.ckpt', global_step=max_steps)
    projector_writer.close()
    session.close()

    # with tf.Session() as session:
    #     session.run(init)
    #     writer = tf.summary.FileWriter('logs/', session.graph)
    #     for epoch in range(10):
    #         session.run(tf.assign(lr, 0.001 * (0.95 ** epoch)))
    #         for batch in range(n_batch):
    #             batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    #             # session.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
    #
    #             summary, _ = session.run([merged, train_step], feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.7})
    #         writer.add_summary(summary, epoch)
    #         test_acc = session.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
    #         train_acc = session.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels, keep_prob: 1.0})
    #         learning_rate = session.run(lr)
    #         print("iter" + str(epoch) + ",Testing Accurary " + str(test_acc) + ",Training Accurary " + str(
    #             train_acc) + "， Learning Rate=" + str(learning_rate))


def variable_summaries(var):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # x input tensor of shape [batch,in_height,in_width,in_channels]
    # W filter / kernel tensor of shape {filter_height,filter_width,in_channels,out_channels}
    # strides[0]=strides[3] = 1
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')



def f14():
    mnist = input_data.read_data_sets('/Users/a1800101471/PycharmProjects/python_demo2/tf_demo/MNIST_data', one_hot=True)
    print('aa')
    batch_size = 100
    n_batch = mnist.train.num_examples

    x = tf.placeholder(tf.float32,[None, 784])
    y = tf.placeholder(tf.float32,[None, 10])
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
            for batch in range(n_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                session.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.7})
            acc = session.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
            print ('Iter' + str(epoch) + ',Testing Accuracy=' + str(acc))

    # -------------------------------------------------


if __name__ == '__main__':
    f14()
