# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import mpl_finance as mpf
import numpy as np
import pandas as pd
from matplotlib.pylab import date2num
import matplotlib.ticker as ticker
import time
import pandas_datareader.data as web

# data=pd.read_csv(u'assets/兴业银行.csv',usecols=['date','open','close','high','low','volume'])
# data[data['volume']==0]=np.nan
# data=data.dropna()
# data.sort_values(by='date',ascending=True,inplace=True)
# 原始的csv 读入进来 DataFrame 的 columns 顺序不符合candlestick_ochl 要求的顺序
# columns 的顺序一定是 date, open, close, high, low, volume
# 这样才符合 candlestick_ochl 绘图要求的数据结构
# 下面这个是改变列顺序最优雅的方法


data = web.DataReader('300059.sz', data_source='yahoo', start='9/1/2018', end='1/30/2019')
data = pd.DataFrame(data)
data['date'] = data.index
time, high, low, open, close, volume = data['date'], data['High'], data['Low'], data['Open'], data['Close'], data[
    'Volume']
# data = data[['date', 'Open', 'Close', 'High', 'Low', 'Volume']]
data = data.head(62)

# 生成横轴的刻度名字
date_tickers = data.date.values

weekday_quotes = [tuple([i] + list(quote[1:])) for i, quote in enumerate(data.values)]
# print weekday_quotes

fig, ax = plt.subplots(figsize=(1200 / 72, 480 / 72))


def format_date(x, pos=None):
    if x < 0 or x > len(date_tickers) - 1:
        return ''
    return date_tickers[int(x)]


ax.xaxis.set_major_locator(ticker.MultipleLocator(6))
ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
ax.grid(True)
# fig.autofmt_xdate()

mpf.candlestick_ochl(ax, zip(time, open, high, low, close), colordown='#ff1717', colorup='#53c156', width=0.6)
plt.setp(plt.gca().get_xticklabels(), rotation=30)
plt.show()
