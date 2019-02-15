# -*- coding: utf-8 -*-
import sklearn

import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import SCORERS

"""
数据清洗与预处理
    对缺失值进行填充
        fill_mean 填充均值 数值
        fill_median 填充中位数 数值
        fill_mode 填充众数 数值文字
        fill_constant 填充常量 数值文字
        fill_rfr 填充随机森林
    特征数值化、离散化
        feature_onehot_encode/feature_onehot_decode 【名义特征】独热编码创建哑变量
        label_encode/label_decode 【有序特征】(不能运算)【分类】转【数值】
        feature_binaryzation/  有距(能加减，不能乘除)、比率(能加减乘除) 数值 【二值化】
        feature_bins 有距(能加减，不能乘除)、比率(能加减乘除) 数值【分箱】
    特征标准化、归一化
    筛选有价值的特征
    分析特征之间的相关性
"""


def fit_transform(transformor, data):
    is_reshape = False
    if data.ndim == 1:
        is_reshape = True
        data = data.reshape(-1, 1)
    result = transformor.fit_transform(data)
    return result.reshape(-1) if is_reshape and result.shape[1] == 1 else result


def inverse_transform(transformor, encode_data):
    is_reshape = False
    if encode_data.ndim == 1:
        is_reshape = True
        encode_data = encode_data.reshape(-1, 1)
    result = transformor.inverse_transform(encode_data)
    return result.reshape(-1) if is_reshape and result.shape[1] == 1 else result


def feature_bins(feature, n_bins, encode, strategy):
    est = KBinsDiscretizer(n_bins, encode, strategy)
    return est, fit_transform(est, feature)


def feature_binaryzation(feature, threshold, copy=True):
    binarizer = Binarizer(threshold=threshold, copy=copy)
    return binarizer, fit_transform(binarizer, feature)


def label_encode(label):
    """
    分类标签转化为分类数值
    :param label:
    :return:
    """
    encoder = LabelEncoder()
    return encoder, encoder.fit_transform(label)


def label_decode(encoder, encode_label):
    """
    分类数值转化为分类标签
    :param encoder:
    :param encode_label:
    :return:
    """
    return encoder.inverse_transform(encode_label)


def feature_encode(feature):
    """
    分类特征转化为分类数值
    :param feature:
    :return:
    """
    encoder = OrdinalEncoder()
    return encoder, fit_transform(encoder, feature)


def feature_decode(encoder, encode_feature):
    """
    分类数值转化为分类特征
    :param encoder:
    :param encode_feature:
    :return:
    """
    return inverse_transform(encoder, encode_feature)


def feature_onehot_encode(data):
    """
    独热编码特征
    :param data: 特征数组
    :return:
    """
    encoder = OneHotEncoder(categories='auto')
    return encoder, fit_transform(encoder, data).toarray()


def feature_onehot_decode(encoder, encode_data):
    """
    独热解码
    :param encoder: 独热编码器
    :param encode_data:
    :return:
    """
    return inverse_transform(encoder, encode_data)


def label_binarizer_encode(data):
    """
    独热编码特征
    :param data: 特征数组
    :return:
    """
    encoder = LabelBinarizer()
    return encoder, fit_transform(encoder, data).toarray()


def normalization(data, feature_range=[0, 1]):
    """
    归一化
    :param data:
    :param feature_range:
    :return:
    """
    scaler = MinMaxScaler(feature_range)
    return scaler, scaler.fit_transform(data)


def inverse_normalization(scaler, normalization_data):
    """
    逆归一化
    :param scaler:
    :param normalization_data:
    :return:
    """
    return scaler.inverse_transform(normalization_data)


def standardization(data):
    """
    标准化
    :param data:
    :return:
    """
    scaler = StandardScaler()
    return scaler, scaler.fit_transform(data)


def inverse_standardization(scaler, standardization_data):
    """
    逆标准化
    :param scaler:
    :param standardization_data:
    :return:
    """
    return scaler.inverse_transform(standardization_data)


def fill_mean(data, missing_values=np.nan, copy=True):
    """
    用均值填充缺失值
    此方法只支持二维以上数据集，一维数据用reshape(-1,1)升维
    :param data: 有缺失值的二维以上数据值 DateFrame 或 numpy.ndarray，
    :param missing_values: 缺失值
    :return: 用均值填充缺失值后的新数据集 numpy.ndarray
    """
    return fill(data, missing_values, "mean", copy)


def fill_median(data, missing_values=np.nan, copy=True):
    """
    用中位数填充缺失值
    :param data:
    :param missing_values:
    :param copy:
    :return:
    """
    return fill(data, missing_values, "median", copy)


def fill_mode(data, missing_values=np.nan, copy=True):
    """
    用众数填充缺失值
    :param data:
    :param missing_values:
    :param copy:
    :return:
    """
    return fill(data, missing_values, "most_frequent", copy)


def fill_constant(data, missing_values=np.nan, fill_value=0, copy=True):
    """
    用常量填充缺失值
    :param data: 有缺失值的二维以上的数据值 DateFrame 或 numpy.ndarray
    :param missing_values: 缺失值
    :param fill_value: 填充值
    :return: 用均值填充缺失值后的新数据集 numpy.ndarray
    """
    return fill(data, missing_values, "constant", copy, fill_value)


def fill(data, missing_values, strategy, copy, fill_value=None):
    imputer = SimpleImputer(missing_values=missing_values, strategy=strategy, copy=copy, fill_value=fill_value)
    return fit_transform(imputer, data)


def fill_rfr(data, target):
    """
    随机森林填充缺失值得新数据集
    :param data: 有缺失值的数据值 numpy.ndarray
    :param target: 标签集 numpy.ndarray
    :return: 用随机森林填充缺失值得新数据集 numpy.ndarray
    """
    isnan_cols = np.isnan(data).sum(axis=0)
    col_isnull_size = np.argsort(isnan_cols)
    X_missing_reg = data.copy()
    target = pd.DataFrame(target)
    for col_index in col_isnull_size:
        if isnan_cols[col_index] == 0:
            continue
        X = pd.DataFrame(X_missing_reg)
        y = X.iloc[:, col_index]
        X = X.iloc[:, X.columns != col_index]
        X = pd.concat([X, target], axis=1)
        X = fill_constant(X, fill_value=0)

        y_train = y[y.notnull()]
        y_test = y[y.isnull()]
        X_train = X[y_train.index, :]
        X_test = X[y_test.index, :]
        rfr = RandomForestRegressor(n_estimators=100)
        rfr.fit(X_train, y_train)
        y_predict = rfr.predict(X_test)
        X_missing_reg[y_test.index, col_index] = y_predict
    return X_missing_reg


def fill_random_forest(data, target, fill_feature, copy=True):
    X = pd.concat([data, target], axis=1)
    y = X[fill_feature]
    X = X.iloc[:, X.columns != fill_feature]
    y_train = y[y.notnull()]
    y_test = y[y.isnull()]
    X_train = X.iloc[y_train.index, :]
    X_test = X.iloc[y_test.index, :]
    rfr = RandomForestRegressor(n_estimators=100)
    rfr.fit(X_train, y_train)
    y_predict = rfr.predict(X_test)
    if not copy:
        data.loc[data.loc[:, fill_feature].isnull(), fill_feature] = y_predict
    return y_predict


def generateNan(data, missing_rate=0.5):
    shape = data.shape
    data_row_len = shape[0]
    data_col_len = shape[1]
    missing_rate = missing_rate
    missing_count = int(np.floor(data_row_len * data_col_len * missing_rate))
    rng = np.random.RandomState(0)
    na_row_indexs = rng.randint(0, data_row_len, missing_count)
    na_col_indexs = rng.randint(0, data_col_len, missing_count)

    data_nan = data.copy()
    data_nan[na_row_indexs, na_col_indexs] = np.nan
    return data_nan


def testFillMissing():
    boston = load_boston()

    # print boston.keys()
    # print boston.filename
    # print boston.feature_names
    # print boston.DESCR

    X_full = boston.data
    y_full = boston.target

    X_missing = generateNan(boston.data)
    X_missing_mean = fill_mean(X_missing)
    X_missing_zero = fill_constant(X_missing, fill_value=0)
    X_missing_reg = fill_rfr(X_missing, y_full)

    for x in [X_full, X_missing_mean, X_missing_zero, X_missing_reg]:
        rfr = RandomForestRegressor(n_estimators=100)
        scores = cross_val_score(rfr, x, y_full, cv=3, scoring='neg_mean_squared_error')
        print scores.mean()


def testTitanic():
    df = pd.read_csv('titanic_2.csv', index_col=0)

    # 填充【均值】
    fill_median(df['Age'].values, copy=False)
    # 填充【众数】
    fill_mode(df['Embarked'].values, copy=False)
    print df.info()
    print df['Embarked'].isnull().sum()

    # 特征【文字】转【数字】
    encoder, df['Sex'] = feature_encode(df['Sex'].values)
    print df.head(), '\n', encoder.categories_

    # 特征【数字】还原【文字】
    df['Sex'] = feature_decode(encoder, df['Sex'].values)
    print df.head()

    # 标签【文字】转【数字】
    encoder, df['Survived'] = label_encode(df['Survived'].values)
    print df.head(), '\n', encoder.classes_

    # 标签【数字】还原【文字】
    df['Survived'] = label_decode(encoder, df['Survived'].values)
    print df.head()

    # 特征【文字】转【01数组】
    encoder, df_onehot = feature_onehot_encode(df['Embarked'].values)
    print df_onehot, encoder.get_feature_names()

    # 特征【01数组】还原【文字】
    df['Embarked'] = feature_onehot_decode(encoder, df_onehot)
    print df.head()

    # 二值化
    binarizer, df['Age'] = feature_binaryzation(df['Age'].values, 30)
    print df.head()

    # 分箱
    est, df['Age'] = feature_bins(df['Age'].values, n_bins=3, encode='ordinal', strategy='uniform')
    print df.head()

    # print df.info()
    # print df.head()
    # encoder, df_onehot = onehot_encode(df[['Sex', 'Embarked']])
    # print pd.DataFrame(df_onehot), '\n', encoder.get_feature_names()
    #
    # df_decode_onehot = onehot_decode(encoder, df_onehot)
    # print pd.DataFrame(df_decode_onehot)


def testNormalization():
    data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
    data = np.array(data)
    scaler, normalization_data = normalization(pd.DataFrame(data), feature_range=[5, 10])
    print data.min(axis=0), data.max(axis=0)
    print scaler.data_min_, scaler.data_max_, scaler.data_range_
    print normalization_data
    data = inverse_normalization(scaler, normalization_data)
    print data


def testStandardization():
    data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
    data = np.array(data)

    scaler, standardization_data = standardization(data)
    print data.mean(axis=0), data.var(axis=0)
    print scaler.mean_, scaler.var_
    print standardization_data.mean(), standardization_data.var()
    data = inverse_standardization(scaler, standardization_data)
    print data


if __name__ == '__main__':
    # for scorer in sorted(SCORERS.keys()):
    #     print scorer
    # testFillMissing()
    # testStandardization()
    # testNormalization()
    testTitanic()
