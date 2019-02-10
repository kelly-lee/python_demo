# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectFromModel

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


def select_by_model(estimator, X, y, threshold):
    select_model = SelectFromModel(estimator, threshold=threshold, norm_order=1)
    return select_model, select_model.fit_transform(X, y)


def select_by_var_threshold(data, threshold):
    """
    方差过滤
    :param data:
    :param threshold:
    :return:
    """
    selector = VarianceThreshold(threshold)
    return selector.fit_transform(data)


def select_by_var_median(data):
    return select_by_var_threshold(data, np.median(data.var()))


def select_by_kbest(data, target, score_func, k):
    """

    :param data:
    :param target:
    :param score_func:
    :param k:
    :return:
    """
    selector = SelectKBest(score_func, k)
    return selector.fit_transform(data, target)


def select_by_kbest_chi2(data, target, k=0):
    """
    卡方过滤
    :param data:
    :param target:
    :param k:
    :return:
    """
    c, p = chi2(data, target)
    if k == 0:
        k = c.shape[0] - (p > 0.05).sum()
    return select_by_kbest(data, target, chi2, k)


def select_by_kbest_fclass(data, target, k=0):
    """
    F检验分类过滤
    :param data:
    :param target:
    :param k:
    :return:
    """
    f, p = f_classif(data, target)
    if k == 0:
        k = f.shape[0] - (p > 0.05).sum()
    return select_by_kbest(data, target, f_classif, k)


def select_by_kbest_freg(data, target, k=0):
    """
    F检验回归过滤
    :param data:
    :param target:
    :param k:
    :return:
    """
    f, p = f_regression(data, target)
    if k == 0:
        k = f.shape[0] - (p > 0.05).sum()
    return select_by_kbest(data, target, f_regression, k)


def select_by_kbest_mic(data, target, k=0):
    """
    互信息分类过滤
    :param data:
    :param target:
    :param k:
    :return:
    """
    mi = mutual_info_classif(data, target)
    if k == 0:
        k = mi.shape[0] - (mi <= 0).sum()
    return select_by_kbest(data, target, mutual_info_classif, k)


def select_by_kbest_mir(data, target, k=0):
    """
    互信息回归过滤
    :param data:
    :param target:
    :param k:
    :return:
    """
    mi = mutual_info_regression(data, target)
    if k == 0:
        k = mi.shape[0] - (mi <= 0).sum()
    return select_by_kbest(data, target, mutual_info_regression, k)


def score(X, y):
    rfc = RandomForestClassifier(n_estimators=10, random_state=0)
    return cross_val_score(rfc, X, y, cv=5).mean()


if __name__ == '__main__':
    df = pd.read_csv('digit_recognizor.csv')
    print df.head()
    X = df.iloc[:, 1:-1].values
    y = df.iloc[:, 0].values
    # print X.shape, score(X, y)

    X_var = select_by_var_threshold(X, 0)
    # print X_var.shape, score(X_var, y)

    # X_chi2 = select_by_kbest_chi2(X_var, y, 450)
    # print X_chi2.shape, score(X_chi2, y)
    #
    # X_f = select_by_kbest_fclass(X_var, y, 0)
    # print X_f.shape, score(X_f, y)

    X_f = select_by_kbest_mic(X_var, y, 0)
    print X_f.shape, score(X_f, y)

    # scores = []
    # for k in range(700, 300, -50):
    #     X_chi2 = select_by_kbest_chi2(X_var, y, k)
    #     s = score(X_chi2, y)
    #     print s
    #     scores.append(s)
    #
    # plt.plot(range(700, 300, -50), scores)
    # plt.show()
