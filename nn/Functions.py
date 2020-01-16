#!/usr/bin/env python
# -*- coding: utf-8 -*-
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class Metrics():
    # https://en.wikipedia.org/wiki/Confusion_matrix
    def __init__(self, tp, fp, fn, tn):
        self.TP = tp
        self.FP = fp
        self.FN = fn
        self.TN = tn
        self.P = self.TP + self.FN  # 真真+假真
        self.N = self.FP + self.TN  # 真假+假假
        self.T = self.TP + self.FP
        self.F = self.FN + self.TN

        self.recall = self.TPR = 1.0 * self.TP / self.P  # 敏感度 实际为【真】的里面预测【正确】【比例】
        self.FNR = 1.0 * self.FN / self.P  # 丢失率

        self.FPR = 1.0 * self.FP / self.N  # 假阳率 实际为【假】的里面预测【错误】【比例】
        self.TNR = self.SPC = 1.0 * self.TN / self.N  # 特异性，真阴率

        self.precision = self.PPV = 1.0 * self.TP / self.T  # 精确率
        self.FDR = 1.0 * self.FP / self.T  # 伪发现率
        self.FOR = 1.0 * self.FN / self.F
        self.NPV = 1.0 * self.TN / self.F  # 阴性预测值

        self.LR_PLUS = self.TPR / self.FPR
        self.LR_SUB = self.FNR / self.TNR
        self.DOR = self.LR_PLUS / self.LR_SUB
        self.total = self.P + self.N
        self.accuracy = 1.0 * (self.TP + self.TN) / self.total  # 准确率
        self.F1 = 2 * (self.precision * self.recall) / (self.precision + self.recall)
        # self.recall_tp = 1.0 * self.tp / self.p  # 实际为【真】的里面预测【正确】【比例】
        # self.recall_tn = 1.0 * self.tn / self.p  # 实际为【假】的里面预测【正确】【比例】
        # self.precision_tp = 1.0 * self.tp / self.t
        # self.precision_f = 1.0 * self.fn / self.f

        # self.f_measure = 2 / (1 / self.precision + 1 / self.recall)


def describe(self):
    data = [
        [self.TP, self.FP, self.T, self.PPV, self.FDR],
        [self.FN, self.TN, self.F, self.FOR, self.NPV],
        [self.P, self.N, self.total, self.accuracy, 1 - self.accuracy],
        [self.TPR, self.FPR, np.nan, self.LR_PLUS, self.DOR],
        [self.FNR, self.SPC, np.nan, self.LR_SUB, self.F1],
    ]
    describe_data = pd.DataFrame(data, index=['true', 'false', 'total', 'recall', 'rate'],
                                 columns=['pos', 'neg', 'total', 'precision', 'accuracy'])
    return describe_data


def normalize_inputs(x):
    m = len(x)
    lambd = 1. / m * np.sum(x, axis=0)
    x = x - lambd
    sigma_square = 1. / m * np.sum(x ** 2, axis=0)
    x = x / sigma_square
    return x


def xavier_init(n_l_1, n_l):
    return np.random.randn(n_l_1, n_l) * np.sqrt(1 / n_l_1)


def msra_init(n_l_1, n_l):
    return np.random.randn(n_l_1, n_l) * np.sqrt(2 / n_l_1)


#################激活函数###################
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
    return x_e / x_e_s


# y是一维分类，yhat是分类onehot
def softmax_loss(y, yhat):
    n = len(y)
    corect_logprobs = -np.log(yhat[range(n), y])
    return np.sum(corect_logprobs)


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


#################正则###################
def l1(theta):
    return np.sum(np.abs(theta))


def l2(theta):
    return np.sum(theta ** 2)


def lf(theta):
    return np.sum(theta ** 2)


def dropout(a, keep_prob=0.8):
    d = np.random.rand(a.shape[0], a.shape[1]) < keep_prob
    a = np.multiply(a, d)
    a /= keep_prob


#################优化###################

# 速度快，会走弯路，也容易走过头
def momentum(theta, m, g, alpha, beta=0.9):
    m = beta * m + (1 - beta) * g
    theta += -alpha * m
    return theta, m


def rms(a, epsilon=1e-08):
    return np.sqrt(a) + epsilon


# 不会出现学习率越来越低的问题，自己调节学习率
def rmsprop(theta, n, g, alpha=0.001, beta=0.999, epsilon=1e-08):
    n = beta * n + (1 - beta) * (g ** 2)
    theta += -alpha * g / rms(n, epsilon)
    return theta, n


# 样本出现越多，学习率越小，样本出现越少，
# 优势学习率越大，自动调节学习率，缺点 迭代次数增多，学习率越来越低，最终趋近于0
def adagrad(theta, n, g, alpha=0.01, epsilon=1e-06):
    n += np.square(g)
    theta += -alpha * g / rms(n, epsilon)
    return theta, n


# 不需要学习率，最快到达终点！！！！！！！！！！！！
def adadelta(theta, e_g2, e_dx2, g, alpha=1.0, rho=0.95, epsilon=1e-06):
    e_g2 = rho * e_g2 + (1 - rho) * (g ** 2)
    dx = - rms(e_dx2, epsilon) / rms(e_g2, epsilon) * g
    # theta += dx
    theta += alpha * dx  # keras 增加了alpha，原论文没有乘alpha，所以alpha默认值是1.0
    e_dx2 = rho * e_dx2 + (1 - rho) * (dx ** 2)
    return theta, e_g2, e_dx2


#
def adam(theta, m, v, g, t, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08):
    t += 1
    m = beta1 * m + (1 - beta1) * g  # momentum 衰减的梯度
    v = beta2 * v + (1 - beta2) * (g ** 2)  # rmsprop 衰减的平方梯度
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    theta += -alpha * m_hat / rms(v_hat, epsilon)
    return theta, m, v, t


def adamax(theta, m, u, g, t, alpha=0.002, beta1=0.9, beta2=0.999):
    t += 1
    m = beta1 * m + (1 - beta1) * g  # momentum
    u = np.maximum(beta2 * u, np.abs(g))  # 比较2个参数大小用maximum，而不是max
    theta += -alpha / (1 - beta1 ** t) * m / u
    # theta += -alpha  * m / (u + epsilon) # keras去掉了1 - beta1 ** t，增加了epsilon
    return theta, m, u, t


# 还没有测试
def nadam(theta, m, m_schedule, n, g, t, alpha=0.002, beta1=0.9, beta2=0.999, epsilon=1e-08, schedule_decay=0.004):
    t = t + 1
    momentum_cache_t = beta1 * (1. - 0.5 * (0.96 ** (t * schedule_decay)))
    momentum_cache_t_1 = beta1 * (1. - 0.5 * (0.96 ** ((t + 1) * schedule_decay)))
    m_schedule_new = m_schedule * momentum_cache_t
    m_schedule_next = m_schedule * momentum_cache_t * momentum_cache_t_1
    m_schedule = m_schedule_new

    g_hat = g / (1 - m_schedule_new)
    m = beta1 * m + (1 - beta1) * g
    m_hat = m / (1 - m_schedule_next)
    n = beta2 * n + (1 - beta2) * (g ** 2)
    n_hat = n / (1 - beta2 ** t)
    m_bar = (1 - momentum_cache_t) * g_hat + (momentum_cache_t_1 * m_hat)
    theta += - alpha * m_bar / rms(n_hat, epsilon)
    return theta, m, m_schedule, n, t


def get_tprs_fprs(y, y_predict):
    tprs = []
    fprs = []
    sorted_predict = y_predict
    # sorted_predict = y_predict[np.argsort(-y_predict)]
    # print sorted_predict
    num = len(sorted_predict) if len(sorted_predict) < 100 else 100
    for thresholds in np.linspace(sorted_predict.max(), sorted_predict.min(), num):
        y_class = (y_predict >= thresholds).astype(int)
        tp = (y & y_class).sum()
        tn = ((1 - y) & (1 - y_class)).sum()
        fn = (y & (1 - y_class)).sum()
        fp = ((1 - y) & y_class).sum()
        tpr = 1.0 * tp / (tp + fn)
        fpr = 1.0 * fp / (fp + tn)
        tprs.append(tpr)
        fprs.append(fpr)
        print
        thresholds, tp, fp, fn, tn, fpr, tpr
    return tprs, fprs


# y_predict = np.array([0.1, 0.4, 0.35, 0.8])
# y = np.array([0, 0, 1, 1])
# https://blog.csdn.net/chekongfu/article/details/86235791
def draw_roc(y, y_predict):
    tprs, fprs = get_tprs_fprs(y, y_predict)
    area = np.trapz(fprs, tprs)
    label = 'ROC (area = %0.2f)' % (area)
    print
    label
    plt.plot(fprs, tprs, label=label)
    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.show()


def draw_ks(y, y_predict):
    tprs, fprs = get_tprs_fprs(y, y_predict)
    # area = np.trapz(fprs, tprs)
    # label = 'ROC (area = %0.2f)' % (area)
    # print label
    plt.plot(range(len(tprs)), tprs, label='tprs')
    plt.plot(range(len(fprs)), fprs, label='fprs')
    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.show()


import matplotlib.pyplot as plt

if __name__ == '__main__':
    # x = np.array([[1, 3], [2, 6], [4, 4]])
    # y = np.sum(x ** 2, axis=1)
    # # print y
    # # print lf(x)
    #
    # x_nor = normalize_inputs(x)
    # plt.scatter(x_nor[:, 0], x_nor[:, 1])
    # # plt.scatter(x[:, 0], x[:, 1])
    # # plt.xlim(-10,10)
    # # plt.ylim(-10,10)
    # plt.show()

    # metrics = Metrics(99713, 45007, 11564, 66233)
    # print metrics.describe()
    print('main')
    y_predict = np.array([0.1, 0.4, 0.35, 0.8])
    y = np.array([0, 0, 1, 1])
    draw_ks(y, y_predict)
