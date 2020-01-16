#!/usr/bin/env python
# -*- coding: utf-8 -*-
# coding: utf-8
import sys

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

from tools import DrawTools
import matplotlib.pyplot   as plt
import seaborn as sns
import numpy as np
from xgboost import XGBClassifier as XGBC

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate, cross_val_score
import lightgbm

"""
序号	属性	说明
1	Age	年龄（数字）
2	Job	工作（工作类型）
3	marital	婚姻状况（绝对：“已婚”，“离婚”，“单身”;注：“离婚”是指离婚或丧偶）
4	education	教育水平（分类：“未知”，“中学”，“小学”，“大专”）
5	default	默认是否有信用（二进制：“是”，“否”）
6	balance	平均每年余额（欧元）（数字）
7	housing	是否有住房贷款（二进制：“是”，“否”）
8	loan	是否有个人贷款（二进制：“是”，“否”）
9	contact	联系人通信类型（分类：“未知”，“电话”，“手机”）
10	day	每个月的最后一个联系日（数字）
11	month	每年的最后一个联系月份
12	duration	上次联系持续时间，以秒为单位（数字）
13	campaign	在此广告系列和此客户中执行的联系数量（数字，包含最后一次联系）
14	pdays	客户最近一次与之前活动联系后经过的天数（数字，-1表示之前未联系过客户）
15	previous	此广告系列和此客户端之前执行的联系数量（数字）
16	poutcome	以前的营销活动的结果（分类：“未知”，“其他”，“失败”，“成功”）
17	y	客户是否订购了定期存款（二进制：“是”，“否”）

调优过程：


encode
train score: 0.927821 
 test score: 0.903867
onehot
train score: 0.937223 
 test score: 0.896133
onehot - del unimportant feature
train score: 0.933905 
 test score: 0.898343
"""


# 标准化
def standard(data):
    from sklearn.preprocessing import StandardScaler
    ss = StandardScaler()
    data[data.dtypes[data.dtypes == 'int64'].index] = ss.fit_transform(data[data.dtypes[data.dtypes == 'int64'].index])
    return data


# 编码
def encode(data):
    data = pd.get_dummies(data, columns=['poutcome', 'contact'])
    data.drop(columns=['poutcome_unknown', 'contact_unknown'], inplace=True)
    data['marital'].replace({'single': 0, 'married': 1, 'divorced': 2}, inplace=True)
    data['education'].replace({'unknown': 0, 'primary': 1, 'tertiary': 2, 'secondary': 3}, inplace=True)
    data['default'].replace({'no': 0, 'yes': 1}, inplace=True)
    data['housing'].replace({'no': 0, 'yes': 1}, inplace=True)
    data['loan'].replace({'no': 0, 'yes': 1}, inplace=True)
    # data['contact'].replace({'unknown': 0, 'cellular': 1, 'telephone': 2}, inplace=True)
    # data['poutcome'].replace({'unknown': 0, 'success': 1, 'failure': 2, 'other': 3}, inplace=True)
    data['month'].replace({'jan': 1
                              , 'feb': 2
                              , 'mar': 3
                              , 'apr': 4
                              , 'may': 5
                              , 'jun': 6
                              , 'jul': 7
                              , 'aug': 8
                              , 'sep': 9
                              , 'oct': 10
                              , 'nov': 11
                              , 'dec': 12
                           }, inplace=True)
    data['y'].replace({'no': 0, 'yes': 1}, inplace=True)
    data['job'] = data.job.astype('category').cat.codes
    data['job'] = data['job'].astype('int64')
    return data


def onehot(data):
    data = pd.get_dummies(data, columns=['job', 'marital', 'education', 'poutcome', 'contact'])
    data['month'].replace({'jan': 1
                              , 'feb': 2
                              , 'mar': 3
                              , 'apr': 4
                              , 'may': 5
                              , 'jun': 6
                              , 'jul': 7
                              , 'aug': 8
                              , 'sep': 9
                              , 'oct': 10
                              , 'nov': 11
                              , 'dec': 12
                           }, inplace=True)
    data['default'].replace({'no': 0, 'yes': 1}, inplace=True)
    data['housing'].replace({'no': 0, 'yes': 1}, inplace=True)
    data['loan'].replace({'no': 0, 'yes': 1}, inplace=True)
    data['y'].replace({'no': 0, 'yes': 1}, inplace=True)
    return data


def train2(data):
    x = data.loc[:, data.columns != 'y']
    y = data['y']
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=100)


def train(data):
    x = data.loc[:, data.columns != 'y']
    y = data['y']
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=100)
    model = XGBC(n_estimators=500, learning_rate=0.05, eval_metric='auc')
    model.fit(xtrain, ytrain)

    # train score: 0.900719
    #  test score: 0.893923
    # train score: 0.920631
    #  test score: 0.899448
    # train score: 0.927821
    #  test score: 0.903867

    ytrain_pred = model.predict(xtrain)
    ytest_pred = model.predict(xtest)
    train_score = accuracy_score(ytrain, ytrain_pred)
    test_score = accuracy_score(ytest, ytest_pred)
    print('train score: %f \n test score: %f' % (train_score, test_score))
    print('roc auc', roc_auc_score(ytrain, ytrain_pred))

    sorted_feature_importances = model.feature_importances_[np.argsort(-model.feature_importances_)]
    feature_importance_names = x.columns[np.argsort(-model.feature_importances_)]
    print([*zip(feature_importance_names, sorted_feature_importances)])
    fi = pd.DataFrame([*zip(feature_importance_names, sorted_feature_importances)], columns=['name', 'score'])
    fi = fi.sort_values(by=['score'], ascending=True)
    fi = fi.reset_index(drop=True)

    ax = plt.gca()
    ax.hlines(y=fi.index, xmin=0, xmax=fi.score, color='firebrick', alpha=0.4, linewidth=30)
    for index, row in fi.iterrows():
        plt.text(row['score'], index, round(row['score'], 2), horizontalalignment='left',
                 verticalalignment='center', fontdict={'color': 'black', 'fontsize': 30})

    plt.yticks(fi.index, fi.name, fontsize=30)
    # ax.scatter(x=fi.index, y=fi.score, s=75, color='firebrick', alpha=0.7)
    plt.show()

    train_confusion_matrix = confusion_matrix(ytrain, ytrain_pred)
    test_confusion_matrix = confusion_matrix(ytest, ytest_pred)
    print('train confusion matrix:\n %s' % train_confusion_matrix)
    print('test confusion matrix:\n %s' % test_confusion_matrix)
    train_classification_report = classification_report(ytrain, ytrain_pred)
    test_classification_report = classification_report(ytest, ytest_pred)
    print('train classification report:\n %s' % train_classification_report)
    print('test classification repor:\n %s' % test_classification_report)
    return model, fi


# 学习曲线
def study(data):
    x = data.loc[:, data.columns != 'y']
    y = data['y']
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=480)
    xticks = np.arange(500, 600, 5)
    train_scores = []
    test_scores = []
    for i in xticks:
        model = XGBC(n_estimators=i, learning_rate=0.05)
        model.fit(xtrain, ytrain)
        ytrain_pred = model.predict(xtrain)
        ytest_pred = model.predict(xtest)
        train_score = accuracy_score(ytrain, ytrain_pred)
        test_score = accuracy_score(ytest, ytest_pred)
        train_scores.append(train_score)
        test_scores.append(test_score)
    # sorted_feature_importances = model.feature_importances_[np.argsort(-model.feature_importances_)]
    test_scores = np.array(test_scores, dtype='float32')
    sorted_test_scores = test_scores[np.argsort(-test_scores)]
    sorted_xtick = xticks[np.argsort(-test_scores)]
    print([*zip(sorted_test_scores, sorted_xtick)])
    plt.plot(xticks, train_scores, label='train')
    plt.plot(xticks, test_scores, label='test')
    plt.legend()
    plt.show()


data = pd.read_csv('bank.csv', sep=';')
print(data.info())
# 7个
# age 年龄 balance 收入
int_cols = data.dtypes[data.dtypes == 'int64']
obj_cols = data.dtypes[data.dtypes == 'object']
print('整数型：%d 个' % int_cols.count())
print('对象型：%d 个' % obj_cols.count())
# data = standard(data)
data = encode(data)

DrawTools.init()


# for obj_col in obj_cols.index:
#     print(data[obj_col].value_counts())

DrawTools.heatmap(data)
plt.legend()
plt.show()

# 构建新特征
# train score: 0.939159
#  test score: 0.912707

# train score: 0.941925
#  test score: 0.918232
g = data.groupby(['month', 'day']).agg({
    'duration': {'max', 'min', 'mean', 'std'}
    # , 'pdays': {'max', 'min', 'mean', 'std'}
    # , 'campaign': {'max', 'min', 'mean', 'std'}
    # , 'previous': {'max', 'min', 'mean', 'std'}
    # , 'poutcome': {'max', 'min', 'mean', 'std'}

}).reset_index()
# g.fillna(g['duration'].mean(),inplace=True)
data = pd.merge(data, g, on=['month', 'day'], how='left')
data[('duration', 'diff')] = data[('duration', 'max')] - data['duration']
# data[('campaign', 'diff')] = data[('campaign', 'mean')] - data['campaign']
# data[('poutcome', 'diff')] = data[('poutcome', 'mean')] - data['poutcome']
# data[('pdays', 'diff')] = data[('pdays', 'mean')] - data['pdays']
# data[('previous', 'diff')] = data[('previous','max')] - data['previous']
# g2 = data.groupby(['housing', 'marital']).agg({
# 'loan': {'max', 'min', 'mean', 'std'}
# }).reset_index()
# data = pd.merge(data, g2, on=['housing', 'marital'], how='left')
# ---------------


DrawTools.displot_mul(data
                      , feature_h='y'
                      , grid=(5, 4))
plt.legend()
plt.show()

DrawTools.boxplot_mul(data,feature_x='y',grid=(5,4),show_strip=False)
plt.show()

train(data)
model, fi = train(data)

data = data.drop(columns=fi.loc[fi.score < 0.01, 'name'], axis=1)

model, fi = train(data)

# ypred = XGBC.predict_proba(xtest)[:,1].reshape(-1,1)
# from nn import Functions
# Functions.draw_roc(np.array(ytest.values), np.array(ypred))
