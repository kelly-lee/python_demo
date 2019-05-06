#!/usr/bin/env python
# -*- coding: utf-8 -*-
# coding: utf-8
import sys
import sklearn.datasets
import matplotlib.pyplot as plt
from nn.Functions import *


class Linear_Regression:

    def __init__(self):
        self.theta_0 = 0
        self.theta_1 = 0

    # 学习算法的解决方案或函数也称为假设
    def hypothesis(self, X):
        return self.theta_0 + self.theta_1 * X

    # 代价函数
    def cost_function(self, yhat, y):
        m = len(y)
        error = yhat - y  # 建模误差
        # 目标便是选择出可以使得建模误差的平方和能够最小的模型参数
        return np.sum(np.square(error)) / (2 * m)

    def theta_1_derivative(self, yhat, y, x):
        m = len(y)
        return np.sum((yhat - y) * x) / m

    def theta_0_derivative(self, yhat, y):
        m = len(y)
        return np.sum((yhat - y)) / m

    # 批量梯度下降
    def batch_gradient_decent(self, X, y, epochs=1000, learning_rate=0.01):
        losses = []
        for epoch in range(epochs):
            yhat = self.hypothesis(X)
            self.theta_1 += -learning_rate * self.theta_1_derivative(yhat, y, X)
            self.theta_0 += -learning_rate * self.theta_0_derivative(yhat, y)

            if epoch % 100 == 0:
                loss = self.cost_function(yhat, y)
                losses.append(loss)
                print (loss)
        return losses

    def predict(self, X):
        return self.hypothesis(X)


if __name__ == '__main__':
    X, y = sklearn.datasets.make_regression(100, 1, 1, noise=10, random_state=0)
    X = X.ravel()
    print (X.shape, y.shape)
    epochs = 1000
    learning_rate = 0.01
    lr = Linear_Regression()
    losses = lr.batch_gradient_decent(X, y, epochs, learning_rate)
    X_test = np.arange(-5, 5, 0.1)
    y_test = lr.predict(X_test)
    f = plt.figure(figsize=(8, 4))
    ax = f.add_subplot(1, 2, 1)
    ax.scatter(X, y)
    ax.plot(X_test, y_test)
    ax = f.add_subplot(1, 2, 2)
    ax.plot(losses)
    plt.show()
