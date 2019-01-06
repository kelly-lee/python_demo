# -*- coding: utf-8 -*-
from talib import abstract
import pandas as pd
import Indicators as ind

import pandas_datareader.data as web
import matplotlib.pyplot as plt
import talib
import numpy as np

# print (talib.abstract.WMA)
# print (talib.abstract.EMA)
# print (talib.abstract.SMA)

# print (talib.abstract.BBANDS)


# print(talib.abstract.MACD)
# print(talib.abstract.MACDEXT)
# print(talib.abstract.MACDFIX)

# print(talib.abstract.CCI)
# print(talib.abstract.RSI)


# print (talib.abstract.EOM)

# print (talib.abstract.ATR)

# print (talib.abstract.PLUS_DI)
# print (talib.abstract.PLUS_DM)
# print (talib.abstract.MINUS_DI)
# print (talib.abstract.MINUS_DM)
# print (talib.abstract.DX)
# print (talib.abstract.ADX)
# print (talib.abstract.ADXR)

# print (talib.abstract.WILLR)
# print talib.abstract.TRIX
# print talib.abstract.STOCHRSI
#
# print talib.abstract.AROON
# print talib.abstract.AROONOSC
# print talib.abstract.MFI
# print talib.abstract.PPO
# print talib.abstract.OBV
#
# print talib.get_functions()

df = pd.DataFrame()
df['A'] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
print df['A'].clip_lower(0)
# df['B'] = ['NAN', 3, 6, 10, 15, 21, 28, 36, 45]
#
# df['B'] = df['B'].shift(1) + df['A']
# print df['B']
# print df[['A']].apply(lambda x: x.shift(2))

# for i in df['A'].index:
#     print df['A'].iloc[0:i + 1].sum()
#
# print df.cumsum()

# print df.rolling(3).apply(lambda x: (3 - (3 - pd.Series(x).idxmax() - 1)) / 3 * 100)

data = web.DataReader('GOOG', data_source='yahoo', start='6/1/2018', end='12/30/2018')
data = pd.DataFrame(data)
high, low, close, volume = data['High'], data['Low'], data['Close'], data['Volume']
print 'load data'
fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(2, 1, 1)

# ax.plot(SMA(close, time_period=5))
# ax.plot(talib.SMA(close, timeperiod=5))

# ax.plot(WMA(close, time_period=5))
# ax.plot(talib.WMA(close, timeperiod=5))

# ax.plot(EMA(close, time_period=12))
# ax.plot(talib.EMA(close, timeperiod=12))

# ax.plot(ind.DEMA(close, time_period=12))
# ax.plot(talib.DEMA(close, timeperiod=12))

# ax.plot(ind.TEMA(close, time_period=12))
# ax.plot(talib.TEMA(close, timeperiod=12))

# ax.plot(ind.TRIMA(close, time_period=12))
# ax.plot(talib.TRIMA(close, timeperiod=12))

# u, m, l = BBANDS(close, 20)
# upper, middle, lower = talib.BBANDS(close, 20)
# ax.plot(u)
# ax.plot(m)
# ax.plot(l)
# ax.plot(upper)
# ax.plot(middle)
# ax.plot(lower)

# m, s, h = MACD(close, fast_period=12, slow_period=26, signal_period=9)
# macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
# ax.plot(m)
# ax.plot(s)
# ax.bar(data.index, h)
# ax.plot(macd)
# ax.plot(macdsignal)
# ax.bar(data.index, macdhist)

# PPO有问题********************
# p, p_s, p_h = PPO(close, fast_period=12, slow_period=26, signal_period=9)
# ppo_hist = talib.PPO(close, fastperiod=12, slowperiod=26)
# ax.plot(p)
# ax.plot(p_s)
# ax.plot(p_h)
# ax.plot(data.index, ppo_hist)

s_k, s_d = ind.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowd_period=3)
# slow_k, slow_d = talib.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowd_period=3)
ax.plot(s_k)
ax.plot(s_d)
# ax.plot(slow_k)
# ax.plot(slow_d)

# f_k, f_d = STOCHF(high, low, close, fastk_period=5, fastd_period=3)
# fast_k, fast_d = talib.STOCHF(high, low, close, fastk_period=5, fastd_period=3)
# ax.plot(f_k)
# ax.plot(f_d)
# ax.plot(fast_k)
# ax.plot(fast_d)

# ax.plot(RSI(close, time_period=14))
# ax.plot(talib.RSI(close, timeperiod=14))

# CCI有差别********************
# ax.plot(CCI(high, low, close, time_period=14))
# ax.plot(talib.CCI(high, low, close, timeperiod=14))

# ax.plot(ROC(close, time_period=10))
# ax.plot(talib.ROC(close, timeperiod=10))

# ax.plot(MFI(high, low, close, volume, time_period=14))
# ax.plot(talib.MFI(high, low, close, volume, timeperiod=14))

# ax.plot(WILLR(high, low, close, time_period=14))
# ax.plot(talib.WILLR(high, low, close, timeperiod=14))

# ax.plot(TRIX(close, time_period=14))
# ax.plot(talib.TRIX(close, timeperiod=14))

# a_u, a_d = AROON(high, low, time_period=14)
# aroon_up, aroon_down = talib.AROON(high, low, timeperiod=14)
# ax.plot(a_u)
# ax.plot(a_d)
# ax.plot(aroon_up)
# ax.plot(aroon_down)

# ax.plot(AROONOSC(high, low, time_period=14))
# ax.plot(talib.AROONOSC(high, low, timeperiod=14))

# ax.plot(EMV(high, low, volume, time_period=14), label="emv")

# ax.plot(ind.ATR(high, low, close, time_period=14))
# ax.plot(talib.ATR(high, low, close, timeperiod=14))

# OBV有差别********************
# ax.plot(OBV(high, volume))
# ax.plot(talib.OBV(high, volume))

# pdm, mdm = ind.DM(high, low, close, time_period=14)
# ax.plot(pdm)
# ax.plot(mdm)
# ax.plot(talib.MINUS_DM(high, low, timeperiod=14))
# ax.plot(talib.PLUS_DM(high, low, timeperiod=14))

# pdi, mdi = ind.DI(high, low, close, time_period=14)
# ax.plot(pdi)
# ax.plot(mdi)
# ax.plot(talib.PLUS_DI(high, low, close, timeperiod=14))
# ax.plot(talib.MINUS_DI(high, low, close, timeperiod=14))

# ax.plot(ind.DX(high, low, close, time_period=14))
# ax.plot(talib.DX(high, low, close, timeperiod=14))
#
# ax.plot(ind.ADXR(high, low, close, time_period=14))
# ax.plot(talib.ADXR(high, low, close, timeperiod=14))

##################################################################################################


# ax.plot(STOCHRSI(data['Close'], time_period=14))
# ax.plot(talib.STOCHRSI(data['Close'], timeperiod=14))


plt.legend()
plt.show()
