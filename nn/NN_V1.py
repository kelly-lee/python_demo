#!/usr/bin/env python
# -*- coding: utf-8 -*-
# coding: utf-8
import sys
import numpy as np
from Functions import *
import sklearn.datasets


def forward_propagation(a_pre, W, b):
    # z1 = X.dot(W1) + b1
    z = np.dot(a_pre, W) + b
    return z


def backward_propagation(dz, a_pre):
    # dW2 = (a1.T).dot(delta2)
    # db2 = np.sum(delta2, axis=0, keepdims=True)
    dw = np.dot(a_pre.T, dz)
    db = np.sum(dz, axis=0)
    return dw, db


def derivative(delta_next, w_next, a):
    # delta1 = delta2.dot(W2.T) * (1 - np.power(a1, 2))
    return np.dot(delta_next, w_next.T) * tanh_derivative(a)


def init_weights(shape):
    return np.random.randn(shape[0], shape[1]) / np.sqrt(shape[0])  # 正态分布
    # return 2 * np.random.random(shape) - 1  # 0中心化随机数


def init_biases(shape):
    return np.zeros((1, shape[1]))


def loss(ws, a2, n, y):
    corect_logprobs = -np.log(a2[range(n), y])
    data_loss = np.sum(corect_logprobs)
    # Add regulatization term to loss (optional)
    data_loss += 0.01 / 2 * (np.sum(np.square(ws[0])) + np.sum(np.square(ws[1])))
    data_loss = 1. / n * data_loss
    return data_loss


def predict(ws, bs, x):
    z1 = forward_propagation(x, ws[0], bs[0])
    a1 = tanh(z1)
    z2 = forward_propagation(a1, ws[1], bs[1])
    a2 = softmax(z2)
    return np.argmax(a2, axis=1)


def train(x, y, times):
    np.random.seed(0)
    n = x.shape[0]
    input_dim = x.shape[1]  # 输入层节点数
    hidden_dim = 3  # 隐藏层节点数
    output_dim = np.unique(y).size  # 输出层节点数

    reg_lambda = 0.01  # 正则强度
    learning_rate = 0.01  # 学习素丽

    dims = [input_dim, hidden_dim, output_dim]
    # W1 = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
    # b1 = np.zeros((1, hidden_dim))
    ws = [init_weights((dims[i], dims[i + 1])) for i in range(len(dims) - 1)]
    bs = [init_biases((dims[i], dims[i + 1])) for i in range(len(dims) - 1)]

    a0 = x
    for time in range(times):
        z1 = forward_propagation(x, ws[0], bs[0])
        a1 = tanh(z1)
        z2 = forward_propagation(a1, ws[1], bs[1])
        a2 = softmax(z2)

        acts = [a0, a1, a2]

        delta2 = a2.copy()  # 注意需要copy！！！！！
        delta2[range(n), y] -= 1

        dw2, db2 = backward_propagation(delta2, a1)
        delta1 = derivative(delta2, ws[1], acts[1])
        dw1, db1 = backward_propagation(delta1, acts[0])

        dws = [dw1, dw2]
        dbs = [db1, db2]

        # ws[1] -= 0.01 * dw2
        # dW2 += reg_lambda * W2
        dws = [dws[i] + reg_lambda * ws[i] for i in range(len(ws))]
        # 梯度下降参数更新
        ws = [ws[i] - learning_rate * dws[i] for i in range(len(ws))]
        bs = [bs[i] - learning_rate * dbs[i] for i in range(len(bs))]

        if time % 100 == 0:
            print time, loss(ws, a2, n, y)

    return ws, bs


if __name__ == '__main__':
    # x = np.arange(-10, 10, 0.1)
    # draw_all_activation_function(x)

    X, y = sklearn.datasets.make_moons(200, noise=0.20, random_state=1)

    ws, bs = train(X, y, 20000)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = predict(ws, bs, np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    plt.show()
