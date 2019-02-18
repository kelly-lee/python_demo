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


class PredictStockPrice:

    def __init__(self, data, sperator, size=60):
        self.min_max_scaler = MinMaxScaler(feature_range=(0, 1))
        self.size = size

        # 训练集
        train_set = data[:sperator]
        # 测试集(多包含sperator前面size个值)
        test_set = data.iloc[len(train_set) - size:]

        X_train, y_train = self.scale(train_set)
        X_test, y_test = self.scale(test_set)

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.y_real = data[sperator:]

    def scale(self, data):
        data_scaled = self.min_max_scaler.fit_transform(data.values)
        X = []
        y = []
        for i in range(self.size, len(data_scaled)):
            # 0~size行0列训练集，size行0列测试集
            X.append(data_scaled[i - self.size:i, 0])
            y.append(data_scaled[i, 0])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (len(X), self.size, 1))
        return X, y

    def fitLSTM(self):
        # 初始化神经网络
        model = Sequential()
        # 用于添加密集连接的神经网络层
        # 用于添加长短期记忆层
        # units=50输出空间的维度
        # return_sequences=True决定是返回输出序列中的最后一个输出，还是返回完整序列
        # input_shape 作为我们训练集的形状
        model.add(LSTM(units=50, return_sequences=True, input_shape=(self.size, 1)))
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
        model.fit(self.X_train, self.y_train, epochs=50, batch_size=32)
        model.save('LSTM.h5')
        return model

    def fitBiLSTM(self):
        model = Sequential()
        model.add(Bidirectional(LSTM(units=50, return_sequences=True, input_shape=(self.size, 1))))
        model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(units=50, return_sequences=True)))
        model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(units=50, return_sequences=True)))
        model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(units=50)))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        model.compile(optimizer='rmsprop', loss='mean_squared_error')
        model.fit(self.X_train, self.y_train, epochs=50, batch_size=32)
        model.save('BiLSTM.h5')
        return model

    def fitGRU(self):
        model = Sequential()
        model.add(GRU(units=50, return_sequences=True, input_shape=(self.size, 1)))
        model.add(Dropout(0.2))
        model.add(GRU(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(GRU(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(GRU(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(self.X_train, self.y_train, epochs=50, batch_size=32)
        model.save('GRU.h5')
        return model

    def predict(self, model):
        if model is None:
            model = load_model('GRU.h5')
        y_predicted = model.predict(self.X_test)
        rmse = math.sqrt(mean_squared_error(self.y_test, y_predicted))
        print rmse
        y_predicted = self.min_max_scaler.inverse_transform(y_predicted)
        y_predicted = pd.DataFrame(y_predicted, index=self.y_real.index)
        plt.plot(self.y_real, color='blue', label='Real Price')
        plt.plot(y_predicted, color='red', label='Predicted Price')
        plt.title('Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    # 加载数据集
    data = web.DataReader('AABA', data_source='yahoo', start='2008-06-01', end='2019-01-31')
    data.rename(
        columns={'Open': 'open', 'Close': 'close', 'High': 'high', 'Low': 'low', 'Volume': 'volume',
                 'Adj Close': 'adj_close'}, inplace=True)
    print data

    predictStockPrice = PredictStockPrice(data[['close']], '2016-01-01')
    # model = predictStockPrice.fitLSTM()
    # model = predictStockPrice.fitBiLSTM()
    # model = predictStockPrice.fitGRU()
    predictStockPrice.predict(model=None)
