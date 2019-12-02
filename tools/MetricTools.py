#!/usr/bin/env python
# -*- coding: utf-8 -*-
# coding: utf-8
import sys
import numpy as np
from sklearn import datasets
from sklearn.metrics import r2_score, auc, roc_auc_score
import pandas  as pd
from sklearn.model_selection import train_test_split


def multioutput_to_weight(multioutput='uniform_average', y_true=None):
    if multioutput == 'uniform_average':  # 等权平均
        return None
    if multioutput == 'variance_weighted':  # 方差加权平均
        return np.var(y_true, axis=0) / np.sum(np.var(y_true, axis=0), axis=0)
    else:
        return multioutput  # 自定义加权平均


def explained_variance_score(y_true, y_pred, multioutput='uniform_average'):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    raw_values = 1 - np.var(y_true - y_pred, axis=0) / np.var(y_true, axis=0)
    if multioutput == 'raw_values':
        return raw_values
    return np.average(raw_values, weights=multioutput_to_weight(multioutput, y_true))


def mean_absolute_error(y_true, y_pred, multioutput='uniform_average'):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    raw_values = np.mean(np.abs(y_true - y_pred), axis=0)
    if multioutput == 'raw_values':
        return raw_values
    return np.average(raw_values, weights=multioutput_to_weight(multioutput, y_true))


def mean_squared_error(y_true, y_pred, multioutput='uniform_average'):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    raw_values = np.mean(np.square(y_true - y_pred), axis=0)
    if multioutput == 'raw_values':
        return raw_values
    return np.average(raw_values, weights=multioutput_to_weight(multioutput, y_true))


def mean_squared_log_error(y_true, y_pred, multioutput='uniform_average'):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    raw_values = np.mean(np.square(np.log1p(y_true) - np.log1p(y_pred)), axis=0)
    if multioutput == 'raw_values':
        return raw_values
    return np.average(raw_values, weights=multioutput_to_weight(multioutput, y_true))


def median_absolute_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.median(np.abs(y_true - y_pred))


def max_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.max(np.abs(y_true - y_pred))


def r2_score(y_true, y_pred, multioutput='uniform_average'):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    raw_values = 1 - np.sum(np.square(y_true - y_pred), axis=0) / np.sum(np.square(y_true - y_true.mean(axis=0)),
                                                                         axis=0)
    if multioutput == 'raw_values':
        return raw_values
    return np.average(raw_values, weights=multioutput_to_weight(multioutput, y_true))


def zero_one_loss(y_true, y_pred, normalize=True):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = (y_true != y_pred)
    if y_true.ndim == 1:
        return np.average(mask) if normalize else np.sum(mask)
    else:
        return np.average(np.any(mask, axis=1)) if normalize else np.sum(np.any(mask, axis=1))


def hinge_loss(y_true, pred_decision):
    # y_true被编码为 +1 和 -1
    # pred_decision 是用 decision_function 预测得到的作为输出的决策
    y_true = np.array(y_true)
    pred_decision = np.array(pred_decision)
    if pred_decision.ndim == 1:
        return np.average(np.maximum(1 + y_true * pred_decision, 0))
    mask = np.ones_like(pred_decision, dtype='bool')
    mask[np.arange(len(y_true)), y_true] = False
    # y_w真实标签预测出的决策
    y_w = pred_decision[~mask]
    # y_t所有其他标签的预测中的最大值
    y_t = np.max(pred_decision[mask].reshape(len(y_true), -1), axis=1)
    return np.average(np.maximum(1 + y_t - y_w, 0))


def log_loss(y_true, y_pred):
    y_true = np.eye(np.unique(y_true).size)[y_true]
    y_pred = np.array(y_pred)
    return np.average(y_true * np.log(y_pred))


def accuracy_score(y_true, y_pred, normalize=True):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = (y_true == y_pred)
    if y_true.ndim == 1:
        return np.average(mask) if normalize else np.sum(mask)
    else:
        return np.average(np.all(mask, axis=1)) if normalize else np.sum(np.all(mask, axis=1))


def confusion_matrix(y_true, y_pred):
    data = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    data = data.groupby(['y_true', 'y_pred']).size().reset_index(name='count')
    n_classes = np.maximum(data.y_true.nunique(), data.y_pred.nunique())
    confusion_matrix_ = np.zeros(shape=(n_classes, n_classes), dtype=np.int)
    for index, row in data.iterrows():
        x = row['y_true'].astype(int)
        y = row['y_pred'].astype(int)
        val = row['count'].astype(int)
        confusion_matrix_[x][y] = val
    return confusion_matrix_


def tp_fp_fn_support(y_true, y_pred):
    confusion_matrix_ = confusion_matrix(y_true, y_pred)
    s = np.sum(confusion_matrix_, axis=1)
    if (len(confusion_matrix_) == 2):
        tn, fp, fn, tp = confusion_matrix_.ravel()
        t = (tn, tp)
    else:
        t = confusion_matrix_.diagonal()
        p = np.sum(confusion_matrix_, axis=0)
        fp = p - t
        fn = s - t
    return t, fp, fn, s


def precision_score(y_true, y_pred, average='binary'):  # tp,fp
    t, fp, fn, support = tp_fp_fn_support(y_true, y_pred)

    if average == 'binary':
        tp = t[1]
        return tp / (tp + fp)

    precision = t / (t + fp)
    precision[np.isnan(precision)] = 0

    if average is None:
        return precision
    if average == 'micro':
        t_sum = np.sum(t)
        fp_sum = np.sum(fp)
        return t_sum / (t_sum + fp_sum)
    if average == 'macro':
        return np.average(precision)
    if average == 'weighted':
        return np.average(precision, weights=support)


def recall_score(y_true, y_pred, average='binary'):
    t, fp, fn, support = tp_fp_fn_support(y_true, y_pred)

    if average == 'binary':
        tp = t[1]
        return tp / (tp + fn)

    recall = t / (t + fn)
    recall[np.isnan(recall)] = 0
    if average is None:
        return recall
    if average == 'micro':
        t_sum = np.sum(t)
        fn_sum = np.sum(fn)
        return t_sum / (t_sum + fn_sum)
    if average == 'macro':
        return np.average(recall)
    if average == 'weighted':
        return np.average(recall, weights=support)


def fbeta_score(y_true, y_pred, average='binary', beta=1):
    t, fp, fn, support = tp_fp_fn_support(y_true, y_pred)

    if average == 'binary':
        tp = t[1]
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (1 + np.square(beta)) * precision * recall / (np.square(beta) * precision + recall)
        return f1

    precision = t / (t + fp)
    recall = t / (t + fn)
    fbeta = (1 + np.square(beta)) * precision * recall / (np.square(beta) * precision + recall)
    fbeta[np.isnan(fbeta)] = 0

    if average is None:
        return fbeta
    if average == 'micro':
        t_sum = np.sum(t)
        fp_sum = np.sum(fp)
        fn_sum = np.sum(fn)
        precision_micro = t_sum / (t_sum + fp_sum)
        recall_micro = t_sum / (t_sum + fn_sum)
        return (1 + np.square(beta)) * precision_micro * recall_micro / (
                np.square(beta) * precision_micro + recall_micro)
    if average == 'macro':
        return np.average(fbeta)
    if average == 'weighted':
        return np.average(fbeta, weights=support)


def f1_score(y_true, y_pred, average='binary'):
    return fbeta_score(y_true, y_pred, average)


def precision_recall_fscore_support(y_true, y_pred):
    tp, fp, fn, support = tp_fp_fn_support(y_true, y_pred)
    precision = tp / (tp + fp)
    precision[np.isnan(precision)] = 0
    recall = tp / (tp + fn)
    recall[np.isnan(recall)] = 0
    f1 = 2 * precision * recall / (precision + recall)
    f1[np.isnan(f1)] = 0
    return precision, recall, f1, support


def matthews_corrcoef(y_true, y_pred):
    t, fp, fn, support = tp_fp_fn_support(y_true, y_pred)
    if len(support) == 2:
        tp, tn = t[1], t[0]
        mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    else:
        s = np.sum(support)
        c = np.sum(t)
        p = t + fp
        mcc = (c * s - np.dot(p, support)) / np.sqrt(
            (np.square(s) - np.dot(p, p)) * (np.square(s) - np.dot(support, support)))
    return mcc


def roc_curve(y_true, y_pred):
    tprs = []
    fprs = []
    thresholds = []
    for threshold in np.linspace(np.min(y_pred), np.max(y_true), np.minimum(len(y_true), 100) - 1):
        y_pred_class = (y_pred >= threshold).astype(int)
        print(y_true, y_pred_class)
        t, fp, fn, support = tp_fp_fn_support(y_true, y_pred_class)
        tp, tn = t[1], t[0]
        tpr = tp / (tp + fn)
        fpr = fp / (tp + fp)
        tprs.append(tpr)
        fprs.append(fpr)
        thresholds.append(threshold)
        # print(thresholds, tp, fp, support, fpr, tpr)
    return tprs, fprs, thresholds


from sklearn.metrics import roc_curve


def test_reg():
    y_true = [3, 2, 7, 1]
    y_pred = [4, 2, 7, 1]
    print(max_error(y_true, y_pred))  # 1

    y_true = [3, -0.5, 2, 7]
    y_pred = [2.5, 0.0, 2, 8]
    print(r2_score(y_true, y_pred))  # 0.9486081370449679
    print(explained_variance_score(y_true, y_pred))  # 0.9571734475374732

    y_true = [3, 5, 2.5, 7]
    y_pred = [2.5, 5, 4, 8]
    print(mean_squared_log_error(y_true, y_pred))  # 0.03973012298459379

    y_true = [3, -0.5, 2, 7]
    y_pred = [2.5, 0.0, 2, 8]
    print(median_absolute_error(y_true, y_pred))  # 0.5
    print(mean_absolute_error(y_true, y_pred))  # 0.5
    print(mean_squared_error(y_true, y_pred))  # 0.375

    y_true = [[0.5, 1], [-1, 1], [7, -6]]
    y_pred = [[0, 2], [-1, 2], [8, -5]]

    print(r2_score(y_true, y_pred, multioutput='raw_values'))  # array([0.96543779, 0.90816327])
    print(r2_score(y_true, y_pred, multioutput='uniform_average'))  # 0.9368005266622779
    print(r2_score(y_true, y_pred, multioutput='variance_weighted'))  # 0.9382566585956416
    print(r2_score(y_true, y_pred, multioutput=[0.3, 0.7]))  # 0.9253456221198156

    print(explained_variance_score(y_true, y_pred, multioutput='raw_values'))  # array([0.96774194, 1.        ])
    print(explained_variance_score(y_true, y_pred, multioutput='uniform_average'))  # 0.9838709677419355
    print(explained_variance_score(y_true, y_pred, multioutput='variance_weighted'))  # 0.9830508474576269
    print(explained_variance_score(y_true, y_pred, multioutput=[0.3, 0.7]))  # 0.9903225806451612

    print(mean_absolute_error(y_true, y_pred, multioutput='raw_values'))  # array([0.5, 1. ])
    print(mean_absolute_error(y_true, y_pred, multioutput='uniform_average'))  # 0.75
    print(mean_absolute_error(y_true, y_pred, multioutput=[0.3, 0.7]))  # 0.85

    print(mean_squared_error(y_true, y_pred, multioutput='raw_values'))  # array([0.41666667, 1.        ])
    print(mean_squared_error(y_true, y_pred, multioutput='uniform_average'))  # 0.7083333333333334
    print(mean_squared_error(y_true, y_pred, multioutput=[0.3, 0.7]))  # 0.825

    y_true = [[0.5, 1], [1, 2], [7, 6]]
    y_pred = [[0.5, 2], [1, 2.5], [8, 8]]
    print(mean_squared_log_error(y_true, y_pred, multioutput='raw_values'))  # [0.00462428 0.08377444]
    print(mean_squared_log_error(y_true, y_pred, multioutput='uniform_average'))  # 0.044199361889160536
    print(mean_squared_log_error(y_true, y_pred, multioutput=[0.3, 0.7]))  # 0.06002939417970032


def test_loss():
    # y_true = [-1, 1, 1]
    # y_pred = [-2.18177944, 2.36355888, 0.09088972]
    # print(zero_one_loss(y_true, y_pred))
    # print(zero_one_loss(y_true, y_pred, normalize=False))
    y_true = [0, 2, 3]
    pred_decision = [[1.27271897, 0.0341701, -0.683807, -1.40168351],
                     [-1.4545164, -0.58122283, -0.37601581, -0.17100692],
                     [-2.36359486, -0.78635381, -0.27341875, 0.23921861]]
    print(hinge_loss(y_true, pred_decision))

    y_true = [0, 0, 1, 1]
    y_pred = [[.9, .1], [.8, .2], [.3, .7], [.01, .99]]
    print(log_loss(y_true, y_pred))


def test_class():
    # y_true = [2, 0, 2, 2, 0, 1]
    # y_pred = [0, 0, 2, 2, 0, 2]
    y_true = [1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4]
    y_pred = [1, 1, 1, 0, 0, 2, 2, 3, 3, 3, 4, 3, 4, 3]
    print("-----------------------------------------------")
    print(confusion_matrix(y_true, y_pred))
    print("-----------------------------------------------")
    # [0.  1.  1.  0.4 0.5]
    # 0.5714285714285714
    # 0.58
    # 0.7999999999999999
    print(precision_score(y_true, y_pred, average=None))
    print(precision_score(y_true, y_pred, average='micro'))
    print(precision_score(y_true, y_pred, average='macro'))
    print(precision_score(y_true, y_pred, average='weighted'))
    print("-----------------------------------------------")
    # [0.         0.6        0.5        0.66666667 0.5]
    # 0.5714285714285714
    # 0.4533333333333333
    # 0.5714285714285714
    print(recall_score(y_true, y_pred, average=None))
    print(recall_score(y_true, y_pred, average='micro'))
    print(recall_score(y_true, y_pred, average='macro'))
    print(recall_score(y_true, y_pred, average='weighted'))
    print("-----------------------------------------------")
    # [0.         0.75       0.66666667 0.5        0.5]
    # 0.5714285714285714
    # 0.4833333333333333
    # 0.6369047619047619
    print(f1_score(y_true, y_pred, average=None))
    print(f1_score(y_true, y_pred, average='micro'))
    print(f1_score(y_true, y_pred, average='macro'))
    print(f1_score(y_true, y_pred, average='weighted'))
    print("-----------------------------------------------")
    # [0.  1.  1.  0.4 0.5]
    # [0.         0.6        0.5        0.66666667 0.5]
    # [0.         0.75       0.66666667 0.5        0.5]
    # [0 5 4 3 2]
    print(precision_recall_fscore_support(y_true, y_pred))
    print("-----------------------------------------------")
    # 0.5714285714285714
    print(accuracy_score(y_true, y_pred))
    # 0.4796320968792721
    print(matthews_corrcoef(y_true, y_pred))


def test_2_class():
    y_pred = [0, 1, 0, 0]
    y_true = [0, 1, 0, 1]
    print(confusion_matrix(y_true, y_pred))
    print(precision_score(y_true, y_pred))
    print(recall_score(y_true, y_pred))
    print(f1_score(y_true, y_pred))
    print(accuracy_score(y_true, y_pred))
    print(matthews_corrcoef(y_true, y_pred))


if __name__ == '__main__':
    test_loss()
    # test_reg()
    # test_loss()
    # test_class()
    # test_2_class()

    # y_true = np.array([0, 0, 1, 1])
    # y_pred = np.array([0.1, 0.4, 0.35, 0.8])
    # # y_true = [1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4]
    # # y_pred = [1, 1, 1, 0, 0, 2, 2, 3, 3, 3, 4, 3, 4, 3]
    # n_classes = np.unique(y_true).size
    # # y_true = np.eye(n_classes)[y_true]
    # # y_pred = np.eye(n_classes)[y_pred]
    #
    y_true = [0, 1, 0, 1]
    y_pred = [1, 1, 1, 0]
    print(roc_curve(y_true, y_pred))

    # iris = datasets.load_iris()
    # X = iris.data
    # y = iris.target
    # y = np.array(y)
    # import xgboost as xgb
    # from xgboost import XGBClassifier
    # from sklearn.metrics import roc_curve
    # import matplotlib.pyplot as plt
    # from sklearn.metrics import auc
    #
    # xgbc = XGBClassifier(objective="multi:softmax")
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # xgbc.fit(X_train, y_train)
    # n_class = np.unique(y).size
    # y_pred = xgbc.predict_proba(X_test)
    # y_true = np.eye(n_class)[y_test]
    #
    # fprs_list = []
    # tprs_list = []
    #
    # for i in range(n_class):
    #     print(y_true[:, i], y_pred[:, i])
    #     fprs, tprs, thresholds = roc_curve(y_true[:, i], y_pred[:, i])
    #     auc = auc(fprs, tprs)
    #     fprs_list.append(fprs)
    #     tprs_list.append(tprs)
    #     plt.plot(fprs, tprs, label='ROC curve of class{0}(area = {1:0.2f})'.format(i + 1, auc(fprs, tprs)))
    # plt.show()
    #
    # fpr_micro, tpr_micro, _ = roc_curve(y_true.ravel(), y_pred.ravel())
    # auc_micro = auc(fpr_micro, tpr_micro)
    # plt.plot(fpr_micro, tpr_micro, '--', lw=2,
    #          label='ROC curve of mirco(area = {1:0.2f})'.format(i + 1, auc_micro))
    #
    # fpr_macro = np.unique(np.concatenate([fprs_list[i] for i in range(n_class)]))
    # mean_tpr = np.zeros_like(fpr_macro)
    # for i in range(n_class):
    #     mean_tpr += np.interp(fpr_macro, fprs_list[i], tprs_list[i])
    # tpr_macro = mean_tpr / n_class
    # auc_macro = auc(fpr_macro, tpr_macro)
    # plt.plot(fpr_macro, tpr_macro, '--', lw=2,
    #          label='ROC curve of macro(area = {1:0.2f})'.format(i + 1, auc_micro))
    # plt.show()
    # from sklearn.metrics import fbeta_score
    from sklearn.metrics import precision_recall_curve
