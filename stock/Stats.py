# -*- coding: utf-8 -*-
from talib import abstract
import pandas as pd
import stock.Indicators as ind

import pandas_datareader.data as web
import matplotlib.pyplot as plt
import talib
import numpy as np

stocks = ['300040.sz', '300241.sz', '300134.sz', '300111.sz', '300059.sz']
# mdi 22~42   29~33  35~38
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(1, 1, 1)
i = 1
for stock in stocks:
    data = web.DataReader(stock, data_source='yahoo', start='1/1/2016', end='1/30/2019')
    data = pd.DataFrame(data)
    high, low, close, volume = data['High'], data['Low'], data['Close'], data['Volume']
    pdi, mdi = ind.DI(high, low, close, time_period=14)
    adx = ind.ADX(high, low, close, time_period=6)
    min = ind.MIN(close, 20)
    max = ind.MAX(close, 50)
    mdi_min = mdi[close == min]
    pdi_min = pdi[close == min]
    adx_min = adx[close == min]
    diff_min = mdi_min - adx_min
    diif_not_min = mdi[close != min] - adx[close != min]
    print stock, diff_min.mean(), diff_min.median()
    print stock, diff_min.round(0).mode()
    ax.scatter(diff_min, np.full((diff_min.count()), 2 * i), s=20, c='b')
    ax.scatter((mdi - pdi)[close != min], np.full(((mdi - pdi)[close != min]).count(), 2 * i + 1), s=20, c='b')
    i = i + 1

plt.legend()
plt.subplots_adjust(hspace=0.1)
plt.show()
