#!/usr/bin/env python
# -*- coding: utf-8 -*-
# coding: utf-8
import sys
import pandas as pd
from scipy.special._ufuncs import boxcox1p
from scipy.stats import skew, boxcox_normmax

numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
object_dtypes = ['object']


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
        if ((data[col].dtypes == 'object') | (len(value_counts) < 20)):
            value_unique_list.append(value_counts.index.values)
        else:
            value_unique_list.append("%.2f - %.2f - %.2f " % (data[col].min(), data[col].mean(), data[col].max()))
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


def fillna_numeric(data, val):
    data.update(data[numeric_columns(data)].fillna(val))


def fillna_object(data, val):
    data.update(data[obj_columns(data)].fillna(val))


def fillna_mode(data, features):
    data.update(data[features].fillna(data[features].mode()))


def boxcox(data):
    skew_features = data[numeric_columns(data)].apply(lambda col_vals: skew(col_vals)).sort_values(ascending=False)
    # print(skew_features)
    high_skew = skew_features[skew_features > 0.5]
    for feature in high_skew.index:
        data[feature] = boxcox1p(data[feature], boxcox_normmax(data[feature] + 1))
