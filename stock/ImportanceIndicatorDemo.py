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


data = web.DataReader('GOOG', data_source='yahoo', start='1/1/2017', end='1/30/2019')
data = pd.DataFrame(data)
high, low, close, open, volume = data['High'], data['Low'], data['Close'], data['Open'], data['Volume']
# print ind.ROC(close)
# print ind.DRCT(close)
# for period in [5, 10, 20, 30, 60, 120]:
#     data['sma_%d' % period] = ind.SMA(close, period)
#     data['ema_%d' % period] = ind.EMA(close, period)
#     data['wma_%d' % period] = ind.WMA(close, period)
#     data['dema_%d' % period] = ind.DEMA(close, period)
#     data['smma_%d' % period] = ind.SMMA(close, period)
#     data['tema_%d' % period] = ind.TEMA(close, period)


# bb
upper_band, middle_band, lower_band = ind.BBANDS(close, 5, 2, 2)
data['upper_band'] = upper_band
data['middle_band'] = middle_band
data['lower_band'] = lower_band
# macd
macd, macdsignal, macdhist = ind.MACD(close, 12, 26, 9)
data['macd'] = macd
data['macd_signal'] = macdsignal
data['macd_hist'] = macdhist
# kdj
slow_k, slow_d = ind.STOCH(high, low, close, fastk_period=9, slowk_period=3, slowd_period=3)
data['slow_k'] = slow_k
data['slow_d'] = slow_d
fast_k, fast_d = ind.STOCHF(high, low, close, fastk_period=9, fastd_period=3)
data['fast_k'] = fast_k
data['fast_d'] = fast_d
# dmi
data['tr'] = ind.TR(high, low, close)
data['atr'] = ind.ATR(high, low, close, 14)
pdm, mdm = ind.DM(high, low, time_period=14)
data['pdm'] = pdm
data['mdm'] = mdm
pdi, mdi = ind.DI(high, low, close, time_period=14)
data['pdi'] = pdi
data['mdi'] = mdi
data['dx'] = ind.DX(high, low, close, time_period=6)
data['adxr'] = ind.ADXR(high, low, close, time_period=6)
# aroon
aroon_up, aroon_down = ind.AROON(high, low, time_period=14)
data['aroon_up'] = aroon_up
data['aroon_down'] = aroon_down
data['aroonosc'] = ind.AROONOSC(high, low, time_period=14)

data['emv'] = ind.EMV(high, low, volume, time_period=14)
data['trix'] = ind.TRIX(close, time_period=14)
data['cci'] = ind.CCI(high, low, close, 14)
data['rsi'] = ind.RSI(close, 14)
data['mfi'] = ind.MFI(high, low, close, volume, time_period=14)
data['willr'] = ind.WILLR(high, low, close, time_period=14)
data['obv'] = ind.OBV(close, volume)
data['vol_roc'] = ind.ROC(volume)
data['roc'] = ind.ROC(close, 5)



# print data
# X = [high, low, close, open, ind.SMA(close, 5), ind.SMA(close, 10), ind.SMA(close, 20)]
# print X
# y = ind.ROC(close)
# y = y[30:]
data = data.iloc[:, 6:].dropna()
sc = MinMaxScaler(feature_range=(0, 1))

X = data.iloc[:, data.columns != "roc"].shift(5).dropna().values
X = sc.fit_transform(X) * 100
y = data.ix[5:, "roc"].astype('int')

importance(X, y, data.columns)


