# !/usr/bin/env python
# -*- coding: utf-8 -*-
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from nn.Functions import *


class NN():
    def __init__(self):
        self.layers = 0
        self.input_dim = None
        self.units = {}
        self.ws = {}  # weights for every layers
        self.bs = {}  # biases for every layers

        self.activations = {}
        self.activation_datas = {}  # 是否不需要存储
        self.optimizer = None
        self.loss = None

        self.deltas = {}  # 是否不需要存储
        self.dws = {}  # derivative weights for every layers 是否不需要存储
        self.dbs = {}  # derivative biases for every layers 是否不需要存储

    def add(self, units, activation=None, input_dim=None):
        if input_dim is not None:
            self.input_dim = input_dim
            self.units[0] = input_dim
        self.layers += 1
        self.units[self.layers] = units
        self.activations[self.layers] = activation

    def init_weights(self, units_pre, units):
        return 2 * np.random.randn(units_pre, units) - 1
        # return np.random.randn(units_pre, units) / np.sqrt(units_pre)  # 正态分布
        # return 2 * np.random.random(shape) - 1  # 0中心化随机数

    def init_biases(self, units):
        return np.zeros((1, units))

    def init_params(self):
        # 设置随机因子
        np.random.seed(0)
        for layer in range(1, self.layers + 1):
            self.ws[layer] = self.init_weights(self.units[layer - 1], self.units[layer])
            self.bs[layer] = self.init_biases(self.units[layer])

    def update_params(self, learning_rate):
        for layer in range(1, self.layers + 1):
            self.ws[layer] += - learning_rate * self.dws[layer]
            self.bs[layer] += - learning_rate * self.dbs[layer]

    def compile(self, optimizer='sgd', loss='mean_squared_error'):
        # 初始化参数
        self.init_params()

    def do_activation(self, layer, z):
        activation = self.activations[layer]
        if activation is None:
            a = z
        elif activation == 'sigmoid':
            a = sigmoid(z)
        elif activation == 'tanh':
            a = tanh(z)
        elif activation == 'relu':
            a = relu(z)
        elif activation == 'softmax':
            a = softmax(z)
        self.activation_datas[layer] = a
        return a

    def forward_propagation(self, x):
        # 预测也调用这个方法，需要重新初始化X
        self.activation_datas[0] = x
        for layer in range(1, self.layers + 1):
            a_pre, w, b = self.activation_datas[layer - 1], self.ws[layer], self.bs[layer]
            # 计算第i层输入
            z = np.dot(a_pre, w) + b
            # 使用激活函数计算第i层输出
            a = self.do_activation(layer, z)
        return a

    def do_actication_derivative(self, layer, a):
        activation = self.activations[layer]
        if activation is None:
            derivative = a
        elif activation == 'sigmoid':
            derivative = sigmoid_derivative(a)
        elif activation == 'tanh':
            derivative = tanh_derivative(a)
        elif activation == 'relu':
            derivative = relu_derivative(a)
        return derivative

    def backward_propagation(self, y):
        for layer in reversed(range(1, self.layers + 1)):
            a = self.activation_datas[layer]
            if layer == self.layers:
                # 如果y分类标签是一维
                if y.shape != a.shape:
                    delta = a.copy()
                    delta[range(len(y)), y] -= 1
                else:
                    # 如果y分类标签是onehot
                    yhat = a
                    delta = yhat - y
            else:
                delta_next, w_next = self.deltas[layer + 1], self.ws[layer + 1]
                a_derivative = self.do_actication_derivative(layer, a)
                delta = a_derivative * np.dot(delta_next, w_next.T)
            a_pre = self.activation_datas[layer - 1]
            dw = np.dot(a_pre.T, delta)
            db = np.sum(delta, axis=0)
            self.deltas[layer] = delta
            self.dws[layer] = dw
            self.dbs[layer] = db

    def predict(self, x):
        yhat = self.forward_propagation(x)
        return np.argmax(yhat, axis=1)

    def cal_loss(self, y, yhat):
        if y.shape != yhat.shape:
            data_loss = softmax_loss(y, yhat)
        else:
            data_loss = softmax_onehot_loss(y, yhat)

        #正则化
        # s = 0
        # for layer in range(1, self.layers + 1):
        #     s += np.sum(np.square(self.ws[layer]))
        # data_loss += 0.01 / 2 * s


        data_loss = 1. / len(y) * data_loss
        return data_loss

    def train(self, x, y, epochs, learning_rate=0.01, reg_lambda=0.01):
        for epoch in range(epochs):
            # 前向传播
            yhat = self.forward_propagation(x)
            # 反向传播 **********************
            self.backward_propagation(y)
            # 正则化
            # for layer in range(1, self.layers + 1):
            #     self.dws[layer] += reg_lambda * self.ws[layer]
            # 更新参数
            self.update_params(learning_rate)
            # 输出损失
            if epoch % 100 == 0:
                print (epoch, self.cal_loss(y, yhat))

        return self.ws, self.bs


def aa():
    X, y = sklearn.datasets.make_moons(200, noise=0.20, random_state=1)
    input_dim = X.shape[1]  # 输入层节点数  2
    hidden_dim = 5  # 隐藏层节点数
    output_dim = np.unique(y).size  # 输出层节点数 2

    nn = NN()
    nn.add(3, input_dim=input_dim, activation=None)
    # nn.add(5, input_dim=input_dim, activation='tanh')
    # nn.add(3, input_dim=input_dim, activation='tanh')
    nn.add(output_dim, activation='softmax')
    nn.compile()
    nn.train(X, y, 20000)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = nn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    plt.show()


def test2():
    # ------------------------------------------------------------
    df = pd.read_csv('W1data.csv')
    print (df.head())

    # ------------------------------------------------------------
    # Get the wine labels
    y = df[['Cultivar 1', 'Cultivar 2', 'Cultivar 3']].values
    # print y.head()
    # Get inputs; we define our x and y here.
    X = df.drop(['Cultivar 1', 'Cultivar 2', 'Cultivar 3'], axis=1)
    print (X.shape ) # (178, 13)
    print (y.shape)  # Print shapes just to check (178, 3)
    X = X.values
    # ------------------------------------------------------------
    np.random.seed(0)
    nn = NN()
    nn.add(5, input_dim=13, activation='tanh')
    nn.add(3, activation='softmax')
    nn.compile()
    nn.train(X, y, 20000)
    # model = initialize_parameters(nn_input_dim=13, nn_hdim=5, nn_output_dim=3)
    # model = train(model, X, y, learning_rate=0.07, epochs=4500, print_loss=True)
    # plt.plot(losses)
    # plt.show()


# 梯队下降、随机梯度下降、
# relu 有问题
# annealing schedule for the gradient descent learning rate
# minibatch gradient descent
if __name__ == '__main__':
    aa()
    # X, y = sklearn.datasets.make_moons(200, noise=0.20, random_state=1)
    # print y
    # b = np.zeros((y.size, np.unique(y).size))
    # b[np.arange(y.size), y] = 1
    # print b
