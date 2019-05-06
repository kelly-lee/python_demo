#!/usr/bin/env python
# -*- coding: utf-8 -*-
# coding: utf-8
import sys
import sklearn.datasets
import matplotlib.pyplot as plt
from nn.Functions import *


class Linear_Regression:

    def __init__(self):
        self.theta = None

    # 学习算法的解决方案或函数也称为假设
    def hypothesis(self, X):
        x0 = np.zeros((len(X), 1))
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
        return np.sum((yhat - y) * x) / m

    # 批量梯度下降
    def batch_gradient_decent(self, X, y, epochs=1000, learning_rate=0.01):
        # 原数据增加x0列，值为1
        x0 = np.ones((len(X), 1))
        X = np.hstack((x0, X))
        # theta初始化为0
        self.theta = np.zeros((1, X.shape[1] + 1))
        losses = []
        for epoch in range(epochs):
            yhat = self.hypothesis(X)
            self.theta += -learning_rate * self.theta_derivative(yhat, y, X)

            if epoch % 10 == 0:
                loss = self.cost_function(yhat, y)
                losses.append(loss)
                print(loss)
        return losses

    def predict(self, X):
        return self.hypothesis(X)


if __name__ == '__main__':
    # a = np.array([[1, 2], [3, 4], [5, 6]])
    # b = np.array([[1], [2]])
    # print(b * a)
    X, y = sklearn.datasets.make_regression(100, 1, 1, noise=20, random_state=0)
    y = y.reshape(-1, 1)
    epochs = 500
    learning_rate = 0.01
    lr = Linear_Regression()
    losses = lr.batch_gradient_decent(X, y, epochs, learning_rate)
    # 预测
    X_test = np.arange(-5, 5, 0.1).reshape(-1, 1)
    y_test = lr.predict(X_test)

    f = plt.figure(figsize=(8, 4))
    ax = f.add_subplot(1, 2, 1)
    ax.scatter(X, y)
    ax.plot(X_test, y_test)
    # --------------------------------------
    ax = f.add_subplot(1, 2, 2)
    for rate in [0.01, 0.03, 0.1, 0.3, 1]:
        lr = Linear_Regression()
        losses = lr.batch_gradient_decent(X, y, epochs, rate)
        ax.plot(losses, label=rate)
        ax.legend()
    plt.show()
