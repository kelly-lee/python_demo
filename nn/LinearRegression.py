#!/usr/bin/env python
# -*- coding: utf-8 -*-
# coding: utf-8
import sys
import sklearn.datasets
import matplotlib.pyplot as plt
import nn.Functions as Functions
from nn.Functions import *


class Linear_Regression:

    def __init__(self):
        self.theta = None

    # 学习算法的解决方案或函数也称为假设
    def hypothesis(self, X):
        # 原数据增加x0列，值为1
        x0 = np.zeros((len(X), 1)) if X.ndim > 1 else 0
        X = np.hstack((x0, X))
        y = np.dot(X, self.theta.T)
        return y

    # 代价函数
    def cost_function(self, yhat, y):
        m = len(y)
        # 目标便是选择出可以使得建模误差的平方和能够最小的模型参数
        return np.sum(np.square(yhat - y)) / (2 * m)

    def theta_derivative(self, yhat, y, x):
        m = len(y)
        # 相乘再求和 就是 点乘
        return np.dot(x.T, yhat - y) / m
        # return np.sum((yhat - y) * x) / m

    # 批量梯度下降
    def batch_gradient_decent(self, X, y, epochs=1000, learning_rate=0.01):
        # theta初始化为0
        self.theta = np.zeros((1, X.shape[1] + 1))
        losses = []
        for epoch in range(epochs):
            yhat = self.hypothesis(X)
            # z这里的X是原数据x，而不是增加x0列的X
            self.theta += -learning_rate * self.theta_derivative(yhat, y, X)

            if epoch % 10 == 0:
                loss = self.cost_function(yhat, y)
                losses.append(loss)
                print(loss, self.theta)
        return losses

    # 随机梯度下降
    def stochastic_batch_gradient_descent(self, X, y, epochs=1000, learning_rate=0.01):
        # theta初始化为0
        self.theta = np.zeros((1, X.shape[1] + 1))
        losses = []
        for epoch in range(epochs):
            for i in range(len(X)):
                _y = y[i]
                _x = X[i]
                yhat = self.hypothesis(_x)
                # z这里的X是原数据x，而不是增加x0列的X
                self.theta += -learning_rate * self.theta_derivative(yhat, _y, _x)

                if i % 100 == 0:
                    loss = self.cost_function(yhat, _y)
                    losses.append(loss)
                    print(loss, self.theta)
        return losses

    # 小批量梯度下降
    def mini_batch_gradient_descent(self, X, y, epochs=1000, learning_rate=0.01, batch_size=10):
        # theta初始化为0
        self.theta = np.zeros((1, X.shape[1] + 1))
        losses = []
        for epoch in range(epochs):
            for batch_index in range(int(len(X) / batch_size)):
                _y = y[batch_index:(batch_index + batch_size)]
                _x = X[batch_index:(batch_index + batch_size)]
                yhat = self.hypothesis(_x)
                # z这里的X是原数据x，而不是增加x0列的X
                self.theta += -learning_rate * self.theta_derivative(yhat, _y, _x)

                if batch_index % 2 == 0:
                    loss = self.cost_function(yhat, _y)
                    losses.append(loss)
                    print(loss, self.theta)
        return losses

    # 动能梯度下降
    def momentum(self, X, y, epochs=1000, learning_rate=0.01, batch_size=10):
        # theta初始化为0
        self.theta = np.zeros((1, X.shape[1] + 1))
        v_theta = np.zeros_like(self.theta)
        losses = []
        for epoch in range(epochs):
            for batch_index in range(int(len(X) / batch_size)):
                y_batch = y[batch_index:(batch_index + batch_size)]
                x_batch = X[batch_index:(batch_index + batch_size)]
                # z这里的X是原数据x，而不是增加x0列的X
                y_hat = self.hypothesis(x_batch)
                d_theta = self.theta_derivative(y_hat, y_batch, x_batch)
                self.theta, v_theta = Functions.momentum(self.theta, v_theta, d_theta, learning_rate)

                if batch_index % 10 == 0:
                    loss = self.cost_function(y_hat, y_batch)
                    losses.append(loss)
                    print(loss)
        return losses

    # 动能梯度下降
    def nesterov_momentum(self, X, y, epochs=1000, learning_rate=0.01, batch_size=10):
        # theta初始化为0
        self.theta = np.zeros((1, X.shape[1] + 1))
        v = np.zeros_like(self.theta)
        momentum = 0.9
        losses = []
        for epoch in range(epochs):
            for batch_index in range(int(len(X) / batch_size)):
                y_batch = y[batch_index:(batch_index + batch_size)]
                x_batch = X[batch_index:(batch_index + batch_size)]
                # z这里的X是原数据x，而不是增加x0列的X
                y_hat = self.hypothesis(x_batch)
                d_theta = self.theta_derivative(y_hat, y_batch, x_batch)
                pre_v = v
                v = momentum * pre_v - learning_rate * self.theta_derivative(y_hat, y_batch, x_batch)
                self.theta += v

                if batch_index % 2 == 0:
                    loss = self.cost_function(y_hat, y_batch)
                    losses.append(loss)
                    print(loss, self.theta)
        return losses

    def nesterov_accelerated_gradient(self, X, y, epochs=1000, learning_rate=0.01, batch_size=10):
        # theta初始化为0
        self.theta = np.zeros((1, X.shape[1] + 1))
        v = np.zeros_like(self.theta)
        momentum = 0.9
        losses = []
        for epoch in range(epochs):
            for batch_index in range(int(len(X) / batch_size)):
                _y = y[batch_index:(batch_index + batch_size)]
                _x = X[batch_index:(batch_index + batch_size)]
                yhat = self.hypothesis(_x)
                # z这里的X是原数据x，而不是增加x0列的X
                pre_v = v
                v = momentum * pre_v - learning_rate * self.theta_derivative(yhat, _y, _x)
                v = v + momentum * (v - pre_v)
                self.theta += v

                if batch_index % 2 == 0:
                    loss = self.cost_function(yhat, _y)
                    losses.append(loss)
                    print(loss, self.theta)
        return losses

    def adagrad(self, X, y, epochs=1000, learning_rate=0.01, batch_size=10):
        # theta初始化为0
        self.theta = np.zeros((1, X.shape[1] + 1))
        a = np.zeros_like(self.theta)  # 梯度累加
        losses = []
        for epoch in range(epochs):
            for batch_index in range(int(len(X) / batch_size)):
                y_batch = y[batch_index:(batch_index + batch_size)]
                x_batch = X[batch_index:(batch_index + batch_size)]
                y_hat = self.hypothesis(x_batch)
                g = self.theta_derivative(y_hat, y_batch, x_batch)

                self.theta, a = Functions.adagrad(self.theta, a, g, learning_rate)

                if batch_index % 10 == 0:
                    loss = self.cost_function(y_hat, y_batch)
                    losses.append(loss)
                    print(loss, self.theta)
        return losses

    def rmsprop(self, X, y, epochs=1000, learning_rate=0.01, batch_size=10):
        # theta初始化为0
        self.theta = np.zeros((1, X.shape[1] + 1))
        s_theta = np.zeros_like(self.theta)
        losses = []
        for epoch in range(epochs):
            for batch_index in range(int(len(X) / batch_size)):
                y_batch = y[batch_index:(batch_index + batch_size)]
                x_batch = X[batch_index:(batch_index + batch_size)]
                y_hat = self.hypothesis(x_batch)
                d_theta = self.theta_derivative(y_hat, y_batch, x_batch)

                self.theta, s_theta = Functions.rmsprop(self.theta, s_theta, d_theta, learning_rate)

                if batch_index % 10 == 0:
                    loss = self.cost_function(y_hat, y_batch)
                    losses.append(loss)
                    print(loss, self.theta)
        return losses

    def adam(self, X, y, epochs=1000, learning_rate=0.001, batch_size=10):
        # theta初始化为0
        self.theta = np.zeros((1, X.shape[1] + 1))
        v_theta = np.zeros_like(self.theta)  # 梯度累加
        s_theta = np.zeros_like(self.theta)
        t = 0
        losses = []
        for epoch in range(epochs):
            for batch_index in range(int(len(X) / batch_size)):
                y_batch = y[batch_index:(batch_index + batch_size)]
                x_batch = X[batch_index:(batch_index + batch_size)]
                y_hat = self.hypothesis(x_batch)
                d_theta = self.theta_derivative(y_hat, y_batch, x_batch)

                self.theta, v_theta, s_theta, t = Functions.adam(self.theta, v_theta, s_theta, d_theta, t,
                                                                 learning_rate)

                if batch_index % 10 == 0:
                    loss = self.cost_function(y_hat, y_batch)
                    losses.append(loss)
                    print(loss, self.theta)
        return losses

    def adamax(self, X, y, epochs=1000, learning_rate=0.001, batch_size=10):
        # theta初始化为0
        self.theta = np.zeros((1, X.shape[1] + 1))
        m = np.zeros_like(self.theta)  # 梯度累加
        u = np.zeros_like(self.theta)
        t = 0
        losses = []
        for epoch in range(epochs):
            for batch_index in range(int(len(X) / batch_size)):
                y_batch = y[batch_index:(batch_index + batch_size)]
                x_batch = X[batch_index:(batch_index + batch_size)]
                y_hat = self.hypothesis(x_batch)

                g = self.theta_derivative(y_hat, y_batch, x_batch)

                self.theta, m, u, t = Functions.adamax(self.theta, m, u, g, t,
                                                       learning_rate)

                if batch_index % 10 == 0:
                    loss = self.cost_function(y_hat, y_batch)
                    losses.append(loss)
        return losses

    def adadelta(self, X, y, epochs=1000, learning_rate=0.01, batch_size=10):
        # theta初始化为0
        self.theta = np.zeros((1, X.shape[1] + 1))
        e_g2 = np.zeros_like(self.theta)
        e_dx2 = np.zeros_like(self.theta)
        losses = []
        for epoch in range(epochs):
            for batch_index in range(int(len(X) / batch_size)):
                y_batch = y[batch_index:(batch_index + batch_size)]
                x_batch = X[batch_index:(batch_index + batch_size)]
                y_hat = self.hypothesis(x_batch)
                g = self.theta_derivative(y_hat, y_batch, x_batch)

                self.theta, e_g2, e_dx2 = Functions.adadelta(self.theta, e_g2, e_dx2, g, learning_rate)

                if batch_index % 10 == 0:
                    loss = self.cost_function(y_hat, y_batch)
                    losses.append(loss)
                    print(loss, self.theta)
        return losses

    def predict(self, X):
        return self.hypothesis(X)


if __name__ == '__main__':
    # a = np.array([[1, 2], [3, 4], [5, 6]])
    # b = np.array([[1], [2]])
    # print(b * a)
    X, y = sklearn.datasets.make_regression(1000, 1, 1, noise=20, random_state=0)
    y = y.reshape(-1, 1)
    epochs = 20
    learning_rate = 0.1
    lr = Linear_Regression()
    # losses = lr.batch_gradient_decent(X, y, epochs, learning_rate)
    # losses = lr.stochastic_batch_gradient_descent(X, y, epochs, learning_rate)
    # losses = lr.mini_batch_gradient_descent(X, y, epochs, learning_rate)
    # losses = lr.momentum(X, y, epochs, learning_rate)
    # losses = lr.nesterov_accelerated_gradient(X, y, epochs, learning_rate)
    # losses = lr.adagrad(X, y, epochs, 2)
    losses = lr.adadelta(X, y, epochs, 5000)
    # losses = lr.RMSprop(X, y, epochs, learning_rate)
    # losses = lr.adam(X, y, epochs, learning_rate)
    # losses = lr.adamax(X, y, epochs, 0.2)

    # 预测
    X_test = np.arange(-5, 5, 0.1).reshape(-1, 1)
    y_test = lr.predict(X_test)

    f = plt.figure(figsize=(8, 4))
    ax = f.add_subplot(1, 2, 1)
    ax.scatter(X, y)
    ax.plot(X_test, y_test)
    # --------------------------------------
    ax = f.add_subplot(1, 2, 2)
    # for rate in [0.01, 0.03, 0.1, 0.3, 1]:
    #     lr = Linear_Regression()
    #     losses = lr.batch_gradient_decent(X, y, epochs, rate)
    ax.plot(losses, label=learning_rate)
    ax.set_ylim(0, 3000)
    ax.legend()
    plt.show()
