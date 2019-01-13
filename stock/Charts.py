# -*- coding: utf-8 -*-

from datetime import datetime

from talib import abstract
import pandas as pd
import stock.Indicators as ind

import pandas_datareader.data as web
import matplotlib.pyplot as plt
import mpl_finance as mpf
import talib
import numpy as np
import matplotlib.dates as mdates
import statsmodels.tsa.stattools as ts

import matplotlib as mpl
import matplotlib.ticker as ticker


def drawK(ax, date_index, high, low, open, close):
    mpf.candlestick_ohlc(ax, zip(date_index, open, high, low, close),
                         colorup='grey', colordown='grey')
    drawDate(ax, date)


def drawBBANDS(ax, price, period=20):
    upper_band, middle_band, lower_band = ind.BBANDS(price, time_period=period)
    ax.plot(upper_band, label='upper_band')
    ax.plot(middle_band, label='middle_band')
    ax.plot(lower_band, label='lower_band')


def drawSMA(ax, price, periods=[5, 10, 20, 30, 60, 120]):
    for period in periods:
        sma = ind.SMA(price, period)
        ax.plot(sma, label='sma%d' % period, linewidth=1)


def drawEMA(ax, price, periods=[5, 10, 20, 30, 60, 120]):
    for period in periods:
        ema = ind.EMA(price, period)
        ax.plot(ema, label='ema%d' % period)


################################################################################################
def drawKDJ(ax, prices, periods=[9, 3, 3], hlines=[-14, -3, 6.5, 17, 95]):
    high, low, close = prices[0], prices[1], prices[2]
    slow_k, slow_d = ind.STOCH(high, low, close, fastk_period=periods[0], slowk_period=periods[1],
                               slowd_period=periods[2])
    # ax.plot(slow_d, label='d')
    # ax.plot(slow_k, label='k')
    ax.plot(3 * slow_k - 2 * slow_d, label='j')
    drawHline(ax, hlines)
    # ax.fill_between(time, 80, rsi, where=rsi >= 80, facecolor='green')
    # ax.fill_between(time, 20, rsi, where=rsi <= 20, facecolor='red')


def drawWR(ax, prices, periods=[6], hlines=[-98, -93, -88, -83.5, -25, -11]):
    high, low, close = prices[0], prices[1], prices[2]
    for period in periods:
        wr = ind.WILLR(high, low, close, time_period=period)
        ax.plot(wr, label='wr%d' % period)
    drawHline(ax, hlines)


def drawDMI(ax, prices, periods=[14, 6], hlines=[10, 12, 16, 20, 22]):
    high, low, close = prices[0], prices[1], prices[2]
    pdi, mdi = ind.DI(high, low, close, time_period=periods[0])
    adx = ind.ADX(high, low, close, time_period=periods[1])
    adxr = ind.ADXR(high, low, close, time_period=periods[1])
    ax.plot(pdi, label='pdi%d' % periods[0])
    # ax.plot(mdi, label='mdi%d' % periods[0])
    # ax.plot(adx, label='adx%d' % periods[1])
    # ax.plot(adxr, label='adxr%d' % periods[1])
    # ax.bar(pdi.index, (pdi - mdi).clip_lower(0), facecolor='r')
    # ax.bar(pdi.index, (pdi - mdi).clip_upper(0), facecolor='g')
    drawHline(ax, hlines)


def drawCCI(ax, prices, periods=[14], hlines=[-231, -138, -110, -83, 50]):
    high, low, close = prices[0], prices[1], prices[2]
    for period in periods:
        cci = ind.CCI(high, low, close, time_period=period)
        ax.plot(cci, label='cci%d' % period)
    drawHline(ax, hlines)


##########################################################################################

def drawRSI(ax, price, periods=[6, 12, 24], hlines=[20, 50, 80]):
    for period in periods:
        rsi = ind.RSI(price, time_period=period)
        ax.plot(rsi, label='rsi%d' % period)
    drawHline(ax, hlines)


def drawMACD(ax, price, periods=[12, 16, 9]):
    macd, macd_signal, macd_histogram = ind.MACD(price, fast_period=periods[0], slow_period=periods[1],
                                                 signal_period=periods[2])
    ax.plot(macd, label='macd%d' % periods[0])
    ax.plot(macd_signal, label='macd_singal%d' % periods[1])
    ax.bar(price.index, macd_histogram.clip_lower(0), facecolor='r')
    ax.bar(price.index, macd_histogram.clip_upper(0), facecolor='g')


def drawHline(ax, hlines):
    for hline in hlines:
        ax.axhline(y=hline, color='grey', linestyle="--", linewidth=1)


def drawDate(ax, date):
    # weeksLoc = mpl.dates.WeekdayLocator()
    # ax.xaxis.set_minor_locator(weeksLoc)
    date_index = np.arange(0, 0 + len(data.index))
    ax.set_xticks(date_index)
    ax.set_xticklabels(ts.date() for ts in date)
    for label in ax.get_xticklabels():
        label.set_visible(False)
    for label in ax.get_xticklabels()[::20]:
        label.set_visible(True)

    # monthsLoc = plt.dates.MonthLocator()
    # ax.xaxis.set_major_locator(monthsLoc)


data = web.DataReader('AAPL', data_source='yahoo', start='1/1/2018', end='1/30/2019')
print ts.adfuller(data['Adj Close'])
data = pd.DataFrame(data)
start = mdates.date2num(data.index.to_pydatetime())[0]
date_index = np.arange(0, 0 + len(data.index))
data['Date'] = data.index
data.index = date_index
date, high, low, open, close, volume = data['Date'], data['High'], data['Low'], data['Open'], data['Close'], data[
    'Volume']

fig = plt.figure(figsize=(16, 8))
ind_size = 2

ax = fig.add_subplot(ind_size, 1, 1)
# drawSMA(ax, close)
# drawEMA(ax, close, periods=[5, 10, 20])

# drawK(ax, date_index, high, low, open, close)
# drawSMA(ax, close)


# drawDMI(ax, [high, low, close])
drawWR(ax, [high, low, close])
drawDate(ax, date)
ax = plt.twinx()
drawKDJ(ax, [high, low, close])
drawDate(ax, date)
# ax.plot(close)
# drawBBANDS(ax, close)

ax = fig.add_subplot(ind_size, 1, 2)
# drawMACD(ax, close)
drawCCI(ax, [high, low, close])
# ax.plot(close, color='grey')
# drawWR(ax, [high, low, close])
# drawDMI(ax, [high, low, close])
drawDate(ax, date)
ax = plt.twinx()
ax.plot(close, color='grey')
drawDate(ax, date)

# ax = plt.twinx()
# drawKDJ(ax, [high, low, close])


# ax = fig.add_subplot(ind_size, 1, 3)

# # ax = plt.twinx()
# drawKDJ(ax, [high, low, close])
# ax = fig.add_subplot(ind_size, 1, 4)
# drawWR(ax, [high, low, close])

plt.legend()
plt.subplots_adjust(hspace=0.1)
# plt.grid(True)
plt.show()
