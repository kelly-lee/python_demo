#!/usr/bin/env python
# -*- coding: utf-8 -*-
# coding: utf-8
import sys
import numpy as np
import pandas as pd
from scipy.special._ufuncs import boxcox1p
from scipy.stats import skew, boxcox_normmax

numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
object_dtypes = ['object']


# Pro01922

def info(data):
    num = data.isnull().sum().sort_values(ascending=False)
    num_p = num / len(data)
    dtypes = data.dtypes
    mode = data.mode().T
    nunique = data.nunique()
    value_unique_list = []
    mode_p_list = []
    for col in data.columns:
        value_counts = data[col].value_counts()
        mode_p = value_counts.iloc[0] / len(data)
        mode_p_list.append(mode_p)
        if ((data[col].dtypes == 'object')):
            if (len(value_counts) < 25):
                value_unique_list.append(value_counts.index.values)
            else:
                value_unique_list.append(value_counts.index.values[:25])
        elif ((data[col].dtypes == 'int64') | (data[col].dtypes == 'float64')):
            if (len(value_counts) < 25):
                value_unique_list.append(value_counts.index.values)
            else:
                value_unique_list.append("%.2f - %.2f - %.2f " % (data[col].min(), data[col].mean(), data[col].max()))
        else:
            value_unique_list.append(' ')
    s_value_unique = pd.Series(value_unique_list, index=data.columns)
    s_mode_p = pd.Series(mode_p_list, index=data.columns)
    info = pd.DataFrame({'null_count': num,
                         'null_p': num_p,
                         'type': dtypes,
                         'mode': mode[0],
                         'mode_p': s_mode_p,
                         'nunique': nunique,
                         'val': s_value_unique})
    info = info.sort_values(by=['type', 'null_p'], ascending=False)
    return info


def numeric_columns(data):
    numerics = []
    for i in data.columns:
        if data[i].dtype in numeric_dtypes:
            numerics.append(i)
    return numerics
    # return data.dtypes[data.dtypes.isin(numeric_dtypes)].index  特别不稳定，随时变，why？？


def obj_columns(data):
    objs = []
    for i in data.columns:
        if data[i].dtype in object_dtypes:
            objs.append(i)
    return objs
    # return data.dtypes[data.dtypes.isin(object_dtypes)].index


def fillna(data, features, val):
    for feature in features:
        data[feature] = data[feature].fillna(val)


def fillna_numeric(data, val):
    fillna(data, numeric_columns(data), val)
    # data.update(data[numeric_columns(data)].fillna(val))


def fillna_object(data, val):
    fillna(data, obj_columns(data), val)
    # data.update(data[obj_columns(data)].fillna(val))


def fillna_mode(data, features):
    for feature in features:
        data[feature] = data[feature].fillna(data[feature].mode()[0])
    # data.update(data[features].fillna(data[features].mode()))


def fillna_mean(data, features):
    for feature in features:
        data[feature] = data[feature].fillna(data[feature].mean())
    # data.update(data[features].fillna(data[features].mean()))


def fillna_median(data, features):
    for feature in features:
        data[feature] = data[feature].fillna(data[feature].median())
    # data.update(data[features].fillna(data[features].median()))


def boxcox(data):
    # 计算数据分布的偏度（skewness）
    skew_features = data[numeric_columns(data)].apply(lambda col_vals: skew(col_vals)).sort_values(ascending=False)
    print(skew_features)
    # 偏度高的进行boxcox转换为正态分布
    # Box和Cox提出的变换可以使线性回归模型满足线性性、独立性、方差齐次以及正态性的同时，又不丢失信息。
    high_skew = skew_features[skew_features > 0.5]
    for feature in high_skew.index:
        if (data[feature].min() >= 0):
            data[feature] = boxcox1p(data[feature], boxcox_normmax(data[feature] + 1))
    return data


# log1p就是log(1+x)，转换数据为高斯分布
def log1p(data):
    return np.log1p(data)


# expm1
def expm1(data):
    return np.expm1(data)


def onehot(data):
    return pd.get_dummies(data).reset_index(drop=True)
