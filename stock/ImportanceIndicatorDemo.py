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

data = ts.get_daily_data(size=300, start_date='20180101', end_date='20190131')
data = data.iloc[:, 7:]
print data.info()
sc = MinMaxScaler(feature_range=(0, 1))

X = data.iloc[:, data.columns != "c_min"].values
# X = sc.fit_transform(X) * 100
y = data.ix[:, "c_min"].astype('int')
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

importance(X, y, data.iloc[:, data.columns != "c_min"].columns)

# Feature ranking:
# 1. feature vol_roc (0.034625)
# 2. feature macd_hist (0.034455)
# 3. feature slow_k (0.033927)
# 4. feature slow_d (0.033756)
# 5. feature pdi (0.033144)
# 6. feature fast_k (0.033110)
# 7. feature mfi (0.032762)
# 8. feature aroon_up (0.032445)
# 9. feature dx (0.032187)
# 10. feature max (0.032034)
# 11. feature trix (0.031859)
# 12. feature atr (0.031802)
# 13. feature emv (0.031757)
# 14. feature fast_d (0.031527)
# 15. feature willr (0.031436)
# 16. feature macd_signal (0.031313)
# 17. feature mdi (0.031262)
# 18. feature middle_band (0.030949)
# 19. feature rsi (0.030854)
# 20. feature cci (0.030788)
# 21. feature mdm (0.030660)
# 22. feature tr (0.030238)
# 23. feature macd (0.030132)
# 24. feature pdm (0.030119)
# 25. feature upper_band (0.030025)
# 26. feature aroonosc (0.029978)
# 27. feature lower_band (0.029535)
# 28. feature obv (0.029305)
# 29. feature c_min (0.028687)
# 30. feature min (0.028619)
# 31. feature aroon_down (0.028507)
# 32. feature adxr (0.028205)
