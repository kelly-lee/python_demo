import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import stock.Indicators as ind

import pandas_datareader.data as web
import matplotlib.pyplot as plt
import talib

from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import MinMaxScaler
import TushareStore as ts


def importance(X, y, columns):
    forest = ExtraTreesClassifier(n_estimators=250,
                                  random_state=0)

    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %s (%f)" % (f + 1, data.columns[indices[f]], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), data.columns[indices])
    plt.xlim([-1, X.shape[1]])
    plt.setp(plt.gca().get_xticklabels(), rotation=60)
    plt.show()


# data = web.DataReader('GOOG', data_source='yahoo', start='1/1/2018', end='1/30/2019')
# data = pd.DataFrame(data)
# high, low, close, open, volume = data['High'], data['Low'], data['Close'], data['Open'], data['Volume']
# data = data.join(ind.ochl2ind(open, close, high, low, volume), how='left')
# # data = data[data['Close'] == data['min']]
# data = data.iloc[:, 6:].dropna()
# data['min'] = data['min'].clip_lower(0.05)

# print data
# X = [high, low, close, open, ind.SMA(close, 5), ind.SMA(close, 10), ind.SMA(close, 20)]
# print X
# y = ind.ROC(close)
# y = y[30:]


# data = ts.get_daily_data(size=300, start_date='20180101', end_date='20190131')
data = ts.query_by_sql("select * from a_daily_sat where date>'2019-04-01' ")
data = data.drop(columns=['pct_sum_next_5', 'id', 'symbol', 'pct_next', 'date'])
data = data.dropna()
print data.info()
# data = data.iloc[:, 7:]

# sc = MinMaxScaler(feature_range=(0, 1))

X = data.iloc[:, data.columns != "pct_sum_next_3"].values
# X = sc.fit_transform(X) * 100
y = np.round(data.ix[:, "pct_sum_next_3"], 0)
# d = sc.inverse_transform(X)
# print data[['c_min', 'vol_roc', 'macd_hist', 'slow_k', 'slow_d', 'pdi']]

# plt.bar(data.index, data['vol_roc'], label='vol_roc')
# plt.bar(np.arange(len(data.index)), data['slow_d'])
# plt.bar(np.arange(len(data.index)), data['c_min'], label='c_min', alpha=0.8)
# plt.plot(data['c_min'], label='c_min', alpha=0.8)
# plt.plot(data['slow_k'])
# plt.plot(data['slow_d'])
# plt.plot(data['mfi'])
# plt.legend()
# plt.show()
y = np.round(data["pct_sum_next_3"].values,0)
print 'begin train'
importance(X, y, data.iloc[:, data.columns != "pct_sum_next_3"].columns)
