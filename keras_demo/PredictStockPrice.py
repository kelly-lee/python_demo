# -*- coding: utf-8 -*-
import os

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# plt.style.use("fivethirtyeight")
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
import math
from sklearn.metrics import mean_squared_error
import pandas_datareader.data as web
from keras.models import load_model


def scale(data, sperator, size):
    # 2016年前的为训练集
    train_set = data[:sperator]
    # 2016年后的为测试集
    y_test = data[sperator:]
    test_set = data.iloc[len(train_set) - size:]

    # 必须是二维数组才能标准化
    sc = MinMaxScaler(feature_range=(0, 1))
    train_set_scaled = sc.fit_transform(train_set.values)
    test_set_scaled = sc.fit_transform(test_set.values)

    # 时间窗口前59个点为特征
    X_train = []
    # 时间窗口第60个点为标签
    y_train = []

    for i in range(size, len(train_set_scaled)):
        # 0~59行0列训练集，60行0列测试集
        X_train.append(train_set_scaled[i - size:i, 0])
        y_train.append(train_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (len(X_train), size, 1))
    # print X_train
    # print y_train

    X_test = []
    for i in range(size, len(test_set_scaled)):
        X_test.append(test_set_scaled[i - size:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (len(X_test), size, 1))
    return sc, X_train, y_train, X_test, y_test

def fitLSTM(X_train, y_train):
    # 初始化神经网络
    model = Sequential()
    # 用于添加密集连接的神经网络层
    # 用于添加长短期记忆层
    # units=50输出空间的维度
    # return_sequences=True决定是返回输出序列中的最后一个输出，还是返回完整序列
    # input_shape 作为我们训练集的形状
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    # 添加防止过度拟合的滤出层
    # 0.2将删除20％的图层
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='rmsprop', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=50, batch_size=32)
    model.save('LSTM.h5')
    return model

def fitBiLSTM(X_train, y_train):
    # 初始化神经网络
    model = Sequential()
    # 用于添加密集连接的神经网络层
    # 用于添加长短期记忆层
    # units=50输出空间的维度
    # return_sequences=True决定是返回输出序列中的最后一个输出，还是返回完整序列
    # input_shape 作为我们训练集的形状
    model.add(Bidirectional(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1))))
    # 添加防止过度拟合的滤出层
    # 0.2将删除20％的图层
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(units=50, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(units=50, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(units=50)))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='rmsprop', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=50, batch_size=32)
    model.save('BiLSTM.h5')
    return model

def fitGRU(X_train, y_train):
    # 初始化神经网络
    model = Sequential()
    # 用于添加密集连接的神经网络层
    # 用于添加长短期记忆层
    # units=50输出空间的维度
    # return_sequences=True决定是返回输出序列中的最后一个输出，还是返回完整序列
    # input_shape 作为我们训练集的形状
    model.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    # 添加防止过度拟合的滤出层
    # 0.2将删除20％的图层
    model.add(Dropout(0.2))
    model.add(GRU(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(GRU(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(GRU(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=50, batch_size=32)
    model.save('GRU.h5')
    return model

def predict(model, sc, X_test, y_test):
    if model is None:
        model = load_model('GRU.h5')
    y_predicted = model.predict(X_test)
    rmse = math.sqrt(mean_squared_error(sc.fit_transform(y_test.values), y_predicted))
    print rmse
    # sc = MinMaxScaler(feature_range=(0, 1))
    y_predicted = sc.inverse_transform(y_predicted)

    predicted = pd.DataFrame(y_predicted, index=y_test.index)
    plt.plot(y_test, color='blue', label='Real Price')
    plt.plot(predicted, color='red', label='Predicted Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()
    print predicted

if __name__ == '__main__':
    # 加载数据集
    data = web.DataReader('AABA', data_source='yahoo', start='2018-06-01', end='2019-01-31')
    data.rename(
        columns={'Open': 'open', 'Close': 'close', 'High': 'high', 'Low': 'low', 'Volume': 'volume',
                 'Adj Close': 'adj_close'}, inplace=True)
    print data

    sc, X_train, y_train, X_test, y_test = scale(data[['close']], '2018-09-01', 60)
    # model = fitLSTM(X_train, y_train)
    # model = fitBiLSTM(X_train, y_train)
    # model = fitGRU(X_train, y_train)
    predict(model=None, sc=sc, X_test=X_test, y_test=y_test)

