from talib import abstract
import pandas as pd
import stock.Indicators as ind

import pandas_datareader.data as web
import matplotlib.pyplot as plt
import talib
import numpy as np


def drawRSI(ax, price, periods=[6, 12, 24], hlines=[20, 50, 80]):
    for period in periods:
        rsi = ind.RSI(price, time_period=period)
        ax.plot(rsi, label='rsi%d' % period)
    drawHline(hlines)
    # ax.fill_between(time, 80, rsi, where=rsi >= 80, facecolor='green')
    # ax.fill_between(time, 20, rsi, where=rsi <= 20, facecolor='red')


def drawMACD(ax, price, periods=[12, 16, 9]):
    macd, macd_signal, macd_histogram = ind.MACD(price, fast_period=periods[0], slow_period=periods[1],
                                                 signal_period=periods[2])
    ax.plot(macd, label='macd')
    ax.plot(macd_signal, label='macd_singal')
    ax.bar(price.index, macd_histogram.clip_lower(0), facecolor='r')
    ax.bar(price.index, macd_histogram.clip_upper(0), facecolor='g')


def drawDMI(ax, prices, periods=[14, 6]):
    high, low, close = prices[0], prices[1], prices[2]
    pdi, mdi = ind.DI(high, low, close, time_period=periods[0])
    dx = talib.DX(high, low, close, time_period=periods[1])
    adxr = talib.ADXR(high, low, close, time_period=periods[1])
    ax.plot(pdi, label='pdi')
    ax.plot(mdi, label='mdi')
    ax.plot(dx, label='dx')
    ax.plot(adxr, label='adxr')


def drawKDJ(ax, prices, periods=[9, 3, 3], hlines=[20, 50, 80]):
    slow_k, slow_d = ind.STOCH(prices[0], prices[1], prices[2], fastk_period=periods[0], slowk_period=periods[1],
                               slowd_period=periods[2])
    ax.plot(slow_d, label='d')
    ax.plot(slow_k, label='k')
    drawHline(hlines)
    # ax.fill_between(time, 80, rsi, where=rsi >= 80, facecolor='green')
    # ax.fill_between(time, 20, rsi, where=rsi <= 20, facecolor='red')


def drawHline(y_arr):
    for y in y_arr:
        ax.axhline(y=y, color='grey', linestyle="--", linewidth=1)


def golden_cross(fast, slow):
    return (fast.shift(1) < slow.shift(1)) & (fast > slow)


def dead_cross(fast, slow):
    return (fast.shift(1) > slow.shift(1)) & (fast < slow)


# AMZN
# GOOG
# .HK .SZ .HZ
# 300059.sz
data = web.DataReader('AAPL', data_source='yahoo', start='5/1/2017', end='1/30/2019')
data = pd.DataFrame(data)
high, low, close, volume = data['High'], data['Low'], data['Close'], data['Volume']

upper_band, middle_band, lower_band = ind.BBANDS(close, time_period=20)
m, s, macd_histogram = ind.MACD(close, fast_period=12, slow_period=26, signal_period=9)
rsi = ind.RSI(close, time_period=14)
min = ind.MIN(close, 20)
max = ind.MAX(close, 50)

df = pd.DataFrame()
df['close'] = close
df['fast'] = m
df['slow'] = s
df['ref'] = df['fast'].shift(1) < df['slow'].shift(1)
df['now'] = df['fast'] > df['slow']
df['cross'] = (m.shift(1) < s.shift(1)) & (m > s)
# print df['cross']
macd_cross = (m.shift(1) < s.shift(1)) & (m > s)
macd_dcross = (m.shift(1) > s.shift(1)) & (m < s)
# print data[macd_cross].Close


size = 2

fig = plt.figure(figsize=(12, 6))

ax = fig.add_subplot(size, 1, 1)
# ax.set_facecolor('grey')

# ax.scatter(data[min == close].index, data[min == close].Close, s=20, c='g')
# ax.scatter(data[max == close].index, data[max == close].Close, s=20, c='r')
ax.scatter(data[golden_cross(m, s)].index, data[golden_cross(m, s)].Close, s=10, c='r')
ax.scatter(data[dead_cross(m, s)].index, data[dead_cross(m, s)].Close, s=10, c='g')
ax.plot(close, color='grey')

# ax.plot(min, color='grey')
# ax.plot(max, color='grey')
# ax.set_xticklabels([])
# plt.grid(True)
# ax.plot(upper_band)
# ax.plot(middle_band)
# ax.plot(lower_band)


ax2 = plt.twinx()
# ax2.bar(data.index, volume, facecolor='grey', alpha=0.5)

# drawMACD(ax2, data.index, m, s, h)

# ax = fig.add_subplot(size, 1, 2)
# ax.bar(data.index, volume)

# ax = fig.add_subplot(size, 1, 2)
# drawMACD(ax, data.index, m, s, h)

ax = fig.add_subplot(size, 1, 2)
# drawDMI(ax, prices=[high, low, close], periods=[14, 6])
drawRSI(ax, close, periods=[6, 12, 24])
# ax.plot(ind.CCI(high, low, close, time_period=20))
# drawMACD(ax, close, periods=[12, 26, 9])
# drawKDJ(ax, prices=[high, low, close], periods=[9, 3, 3])
# ax2.plot(ind.SMA(ind.ROC(volume), 14))
# ax2.plot(ind.ROC(volume, 30))

# slow_k, slow_d = talib.STOCH(high, low, close, fastk_period=9, slowk_period=3, slowd_period=3)
# ax.plot(s_k)
# ax.plot(s_d)
# ax.plot(slow_k)
# ax.plot(slow_d)

plt.legend()
plt.subplots_adjust(hspace=0.1)
# plt.grid(True)
plt.show()
