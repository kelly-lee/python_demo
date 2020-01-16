#!/usr/bin/env python
# -*- coding: utf-8 -*-
# coding: utf-8
import sys

# reload(sys)
# sys.setdefaultencoding('utf8')

# num_round(10)/n_estimators(100) 集成中弱评估器的数量
# slient(false)／slient(true) 训练中是否打印每次训练的结果


from xgboost import XGBRegressor  as XGBR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import LinearRegression as LR
from sklearn.model_selection import KFold, cross_val_score, train_test_split, learning_curve
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error as MSE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from time import time
import datetime


# 简单调用xgboost
def test1():
    data = load_boston()
    X = data.data
    y = data.target
    print(X.shape)
    print(data.data, data.target)
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=0)
    reg = XGBR(n_estimators=100, random_state=0).fit(Xtrain, ytrain)
    print(reg.score(Xtest, ytest))  # R^2
    print(y.mean())
    print(MSE(ytest, reg.predict(Xtest)))
    print(reg.feature_importances_)
    print(data.feature_names[np.argsort(-reg.feature_importances_)])


# 交叉验证xgboost，和rfr，lr对比
def test2():
    data = load_boston()
    X = data.data
    y = data.target
    xgbr = XGBR(n_estimators=100, random_state=0)
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=0)
    xgbr_score = cross_val_score(xgbr, Xtrain, ytrain, cv=5).mean()
    print(xgbr_score)
    xgbr_score = cross_val_score(xgbr, Xtrain, ytrain, cv=5, scoring='neg_mean_squared_error').mean()
    print(xgbr_score)
    print(sorted(sklearn.metrics.SCORERS.keys()))

    rfr = RFR(n_estimators=100, random_state=0)
    rfr_score = cross_val_score(rfr, Xtrain, ytrain, cv=5).mean()
    print(rfr_score)
    rfr_score = cross_val_score(rfr, Xtrain, ytrain, cv=5, scoring='neg_mean_squared_error').mean()
    print(rfr_score)

    lr = LR()
    lr_score = cross_val_score(lr, Xtrain, ytrain, cv=5).mean()
    print(lr_score)
    lr_score = cross_val_score(lr, Xtrain, ytrain, cv=5, scoring='neg_mean_squared_error').mean()
    print(lr_score)


def plot_learning_curve(estimator, title, X, y, ax=None, ylim=None, cv=None, n_jobs=None):
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, random_state=0, n_jobs=n_jobs)
    if ax == None:
        ax = plt.gca()
    else:
        ax = plt.figure()
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel('Training examples')
    ax.set_ylabel('Score')
    ax.grid()
    ax.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color='r', label='Training score')
    ax.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', color='g', label='Test score')
    ax.legend(loc='best')
    return ax


# 学习曲线 看 模型在 训练集和测试集的表现
def test3():
    data = load_boston()
    X = data.data
    y = data.target
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=0)
    estimator = XGBR(n_estimators=100, random_state=0)
    cv = KFold(n_splits=5, shuffle=True, random_state=0)  # 交叉验证模式
    plot_learning_curve(estimator, 'XGB', Xtrain, ytrain, ax=None, cv=cv)
    plt.show()


# 画学习曲线 总泛化误差 方差，偏差
def test4():
    random_state = 420
    data = load_boston()
    X = data.data
    y = data.target
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=random_state)
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)  # 交叉验证模式
    axis = range(300, 500, 10)
    rs = []  # 方差
    vars = []  # 偏差
    ges = []
    for i in axis:
        reg = XGBR(n_estimators=i, random_state=random_state)
        cvs = cross_val_score(reg, Xtrain, ytrain, cv=cv)
        rs.append(cvs.mean())
        vars.append(cvs.var())
        ges.append(1 - cvs.mean() ** 2 + cvs.var())
    max_rs = axis[rs.index(max(rs))]
    min_vars = axis[vars.index(min(vars))]
    min_ges = axis[ges.index(min(ges))]
    print(axis[rs.index(max(rs))], max(rs), vars[rs.index(max(rs))])
    print(axis[vars.index(min(vars))], rs[vars.index(min(vars))], min(vars))
    print(axis[ges.index(min(ges))], rs[ges.index(min(ges))], vars[ges.index(min(ges))])
    plt.figure(figsize=(20, 5))
    plt.plot(axis, np.array(rs) + np.array(vars), c='red', linestyle='-.')
    plt.plot(axis, rs, c='black', label='XGB')
    plt.plot(axis, np.array(rs) - np.array(vars), c='red', linestyle='-.')
    plt.legend()
    plt.show()
    time0 = time()
    print(XGBR(n_estimators=max_rs, random_state=random_state).fit(Xtrain, ytrain).score(Xtest, ytest))
    print(time() - time0)

    time0 = time()
    print(XGBR(n_estimators=min_vars, random_state=random_state).fit(Xtrain, ytrain).score(Xtest, ytest))
    print(time() - time0)

    time0 = time()
    print(XGBR(n_estimators=min_ges, random_state=random_state).fit(Xtrain, ytrain).score(Xtest, ytest))
    print(time() - time0)


# 画取样subsample的学习曲线
def test5():
    random_state = 420
    n_estimators = 380  # test4调完最佳参数是380
    data = load_boston()
    X = data.data
    y = data.target
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=random_state)
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)  # 交叉验证模式
    axis = np.linspace(0.75, 1, 25)
    rs = []  # 方差
    vars = []  # 偏差
    ges = []
    for i in axis:
        reg = XGBR(n_estimators=n_estimators, random_state=random_state, subsample=i)
        cvs = cross_val_score(reg, Xtrain, ytrain, cv=cv)
        rs.append(cvs.mean())
        vars.append(cvs.var())
        ges.append(1 - cvs.mean() ** 2 + cvs.var())
    max_rs = axis[rs.index(max(rs))]
    min_vars = axis[vars.index(min(vars))]
    min_ges = axis[ges.index(min(ges))]
    print(axis)
    print(rs)
    print(axis[rs.index(max(rs))], max(rs), vars[rs.index(max(rs))])
    print(axis[vars.index(min(vars))], rs[vars.index(min(vars))], min(vars))
    print(axis[ges.index(min(ges))], rs[ges.index(min(ges))], vars[ges.index(min(ges))])
    plt.figure(figsize=(20, 5))
    plt.plot(axis, np.array(rs) + np.array(vars), c='red', linestyle='-.')
    plt.plot(axis, rs, c='black', label='XGB')
    plt.plot(axis, np.array(rs) - np.array(vars), c='red', linestyle='-.')
    plt.legend()
    plt.show()
    time0 = time()
    print(XGBR(n_estimators=n_estimators, random_state=random_state, subsample=max_rs).fit(Xtrain, ytrain).score(Xtest,
                                                                                                                 ytest))
    print(time() - time0)

    time0 = time()
    print(
        XGBR(n_estimators=n_estimators, random_state=random_state, subsample=min_vars).fit(Xtrain, ytrain).score(Xtest,
                                                                                                                 ytest))
    print(time() - time0)

    time0 = time()
    print(XGBR(n_estimators=n_estimators, random_state=random_state, subsample=min_ges).fit(Xtrain, ytrain).score(Xtest,
                                                                                                                  ytest))
    print(time() - time0)


def regassess(reg, Xtrain, ytrain, cv, scoring=['r2'], show=True):
    scores = []
    for i in range(len(scoring)):
        score = cross_val_score(reg, Xtrain, ytrain, cv=cv, scoring=scoring[i]).mean()
        print('{}:{:.2f}'.format(scoring[i], score))
        scores.append(score)
    return scores


# learning_rate 学习曲线
def test6():
    random_state = 420
    n_estimators = 380  # test4调完最佳参数是380
    subsample = 0.9
    data = load_boston()
    X = data.data
    y = data.target
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=random_state)
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)  # 交叉验证模式
    axis = np.arange(0.05, 1, 0.05)
    rs = []  # 方差
    vars = []  # 偏差
    ges = []
    for i in axis:
        reg = XGBR(n_estimators=n_estimators, random_state=random_state, subsample=subsample, learning_rate=i)
        cvs = regassess(reg, Xtrain, ytrain, cv=cv, scoring=['r2', 'neg_mean_squared_error'], show=True)[0]
        print(cvs)
        rs.append(cvs.mean())
        vars.append(cvs.var())
        ges.append(1 - cvs.mean() ** 2 + cvs.var())
    max_rs = axis[rs.index(max(rs))]
    min_vars = axis[vars.index(min(vars))]
    min_ges = axis[ges.index(min(ges))]
    print(axis)
    print(rs)
    print(axis[rs.index(max(rs))], max(rs), vars[rs.index(max(rs))])
    print(axis[vars.index(min(vars))], rs[vars.index(min(vars))], min(vars))
    print(axis[ges.index(min(ges))], rs[ges.index(min(ges))], vars[ges.index(min(ges))])
    plt.figure(figsize=(20, 5))
    plt.plot(axis, np.array(rs) + np.array(vars), c='red', linestyle='-.')
    plt.plot(axis, rs, c='black', label='XGB')
    plt.plot(axis, np.array(rs) - np.array(vars), c='red', linestyle='-.')
    plt.legend()
    plt.show()
    time0 = time()
    print(XGBR(n_estimators=n_estimators, random_state=random_state, subsample=max_rs).fit(Xtrain, ytrain).score(Xtest,
                                                                                                                 ytest))
    print(time() - time0)

    time0 = time()
    print(
        XGBR(n_estimators=n_estimators, random_state=random_state, subsample=min_vars).fit(Xtrain, ytrain).score(Xtest,
                                                                                                                 ytest))
    print(time() - time0)

    time0 = time()
    print(XGBR(n_estimators=n_estimators, random_state=random_state, subsample=min_ges).fit(Xtrain, ytrain).score(Xtest,
                                                                                                                  ytest))
    print(time() - time0)


# 测试不同的booster
def test7():
    random_state = 420
    n_estimators = 380  # test4调完最佳参数是380
    subsample = 0.9
    learning_rate = 0.1

    data = load_boston()
    X = data.data
    y = data.target
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=random_state)

    for booster in ['gbtree', 'gblinear', 'dart']:
        reg = XGBR(n_estimators=n_estimators, random_state=random_state, subsample=subsample,
                   learning_rate=learning_rate, booster=booster)
        reg.fit(Xtrain, ytrain)
        print(booster, reg.score(Xtest, ytest))


# objective:
# reg:linear
# binary:logistic
# binary:hinge
# multi:softmax
import xgboost as xgb
from sklearn.metrics import r2_score


# 使用xgboost api，优于sklearn
def test8():
    random_state = 420
    data = load_boston()
    X = data.data
    y = data.target
    Xtrian, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=random_state)
    dtrain = xgb.DMatrix(Xtrian, ytrain)
    dtest = xgb.DMatrix(Xtest, ytest)
    param = {'silent': False,
             'objective': 'reg:linear',
             'eta': 0.1}
    num_boost_round = 380  # n_estimators
    bst = xgb.train(param, dtrain, num_boost_round)
    preds = bst.predict(dtest)
    print(r2_score(ytest, preds))
    print(MSE(ytest, preds))


if __name__ == "__main__":
    test8()
