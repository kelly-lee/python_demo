# -*- coding: utf-8 -*-
import os

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
import math
from sklearn.metrics import mean_squared_error

import pandas_datareader.data as web

#加载数据集
data = web.DataReader('GOOG', data_source='yahoo', start='1/1/2006', end='12/30/2018')
data = pd.DataFrame(data)
print data.head()
print data.info()
training_set = data[:'2016'].iloc[:, 2:3].values


# test_set = data['2017':].iloc[:, 1:2].values
# data['High'][:'2016'].plot(figsize=(16, 4), legend=True)
# data['High']['2017':].plot(figsize=(16, 4), legend=True)
# plt.legend(['2016', '2017'])
# plt.title('GOOG')
# plt.show()

sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)
X_train = []
y_train = []
#2035条记录
for i in range(60, 2268):
    #0~59行0列训练集，60行0列测试集
    X_train.append(training_set_scaled[i - 60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#初始化神经网络
regressor = Sequential()
#用于添加密集连接的神经网络层
#用于添加长短期记忆层
#units=50输出空间的维度
#return_sequences=True决定是返回输出序列中的最后一个输出，还是返回完整序列
#input_shape 作为我们训练集的形状
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
#添加防止过度拟合的滤出层
#0.2将删除20％的图层
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))
regressor.add(Dense(units=1))

regressor.compile(optimizer='rmsprop', loss='mean_squared_error')
regressor.fit(X_train, y_train, epochs=50, batch_size=32)
