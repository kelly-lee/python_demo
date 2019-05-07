#!/usr/bin/env python
# -*- coding: utf-8 -*-
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt


# exp 以自然常数e为底的指数函数
def exp(x):
    return np.exp(x)


# 激活函数 sigmoid
def sigmoid(x):
    return 1 / (1 + exp(-x))


# sigmoid函数梯度
def sigmoid_derivative(sigmoid_x):
    return sigmoid_x * (1 - sigmoid_x)


# 激活函数 tanh
def tanh(x):
    # return 2 * sigmoid(x) - 1
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x))


# tanh 函数梯度
def tanh_derivative(tanh_x):
    return 1 - np.square(tanh_x)


# 激活函数 relu
# relu(x) = max(0,x) = x(x>0)|0(x<=0)
def relu(x):
    return np.where(x < 0, 0, x)


def relu_derivative(x):
    return np.where(x < 0, 0, 1)


def leaky_relu(x):
    return np.where(x < 0, 0.1 * x, x)


def leaky_relu_derivative(x):
    return np.where(x < 0, 0.1, 1)


def elu(x, a):
    return np.where(x <= 0, a * exp(x) - 1, x)


def elu_derivative(x, a):
    return np.where(x <= 0, a * exp(x), 1)


def softmax(x):
    x_e = exp(x - np.max(x))
    x_e_s = np.sum(x_e, axis=1, keepdims=True)
    return x_e / x_e_s + 0.001


# y是一维分类，yhat是分类onehot
def softmax_loss(y, yhat):
    n = len(y)
    corect_logprobs = -np.log(yhat[range(n), y] + 0.001)
    return np.sum(corect_logprobs + 0.001)


# y是onehot，yhat是onehot
def softmax_onehot_loss(y, yhat):
    minval = 0.000000000001
    corect_logprobs = y * -np.log(yhat.clip(min=minval))
    return np.sum(corect_logprobs)


def onehot(y):
    b = np.zeros((len(y), np.unique(y).size))
    b[np.arange(len(y)), y] = 1
    return b


def draw_function(ax, x, label, y, d_label, dx):
    ax.plot(x, y, label=label)
    ax.plot(x, dx, label=d_label)
    ax.legend()


def draw_sigmoid(ax, x):
    draw_function(ax, x, 'sigmoid', sigmoid(x), 'sigmoid_derivative', sigmoid_derivative(sigmoid(x)))


def draw_tanh(ax, x):
    draw_function(ax, x, 'tanh', tanh(x), 'tanh_derivative', tanh_derivative(tanh(x)))


def draw_relu(ax, x):
    draw_function(ax, x, 'relu', relu(x), 'relu_derivative', relu_derivative(x))


def draw_leaky_relu(ax, x):
    draw_function(ax, x, 'leaky_relu', leaky_relu(x), 'leaky_relu_derivative', leaky_relu_derivative(x))


def draw_all_activation_function(x):
    figure = plt.figure()
    ax = figure.add_subplot(3, 2, 1)
    draw_sigmoid(ax, x)
    ax = figure.add_subplot(3, 2, 2)
    draw_tanh(ax, x)
    ax = figure.add_subplot(3, 2, 3)
    draw_relu(ax, x)
    ax = figure.add_subplot(3, 2, 4)
    draw_leaky_relu(ax, x)
    plt.show()


def L1(y, yhat):
    return np.sum(np.abs(y, yhat))
