# -*- coding: utf-8 -*-
import tushare as ts
from sqlalchemy import create_engine
import pandas as pd
import stock.Indicators as ind
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(1, 1, 1)
engine = create_engine('mysql://root:root@127.0.0.1:3306/Stock?charset=utf8')
df = pd.read_sql('h_data', engine)
# df = df.sort_values('date', ascending=True)
# stock = df[df.code == '002660']
# stock.index = stock.date
# high, low, close, volume = stock['high'], stock['low'], stock['close'], stock['volume']
# pdi, mdi = ind.DI(high, low, close, time_period=14)
# print pdi
# ax.plot(pdi)
# plt.show()

# print df.groupby('code')
print df
for code in df['code'].drop_duplicates():
    stock = df[df.code == code]
    stock = stock.sort_values('date', ascending=True)
    stock.index = stock.date
    high, low, close, volume = stock['high'], stock['low'], stock['close'], stock['volume']
    pdi, mdi = ind.DI(high, low, close, time_period=14)
    min = ind.MIN(close, 30)
    pdi_min = pdi[close == min]
    pdi_not_min = pdi[close != min].dropna()
    # ax.scatter(pdi_not_min, np.full((pdi_not_min.count()), code), s=2, c='r', alpha=0.5)
    ax.scatter(pdi_min, np.full((pdi_min.count()), code), s=2, c='b', alpha=0.5)
    print 'stat', code
    # print pdi_min['2019-01-10'].size
    # print pdi_min['2019-01-10'].size != 0
    # print pdi_min
    # try:
    #     if ((pdi_min['2019-01-10'].size != 0)):
    #         pmin = pdi_min['2019-01-10'][0]
    #         print pmin
    #         if ((pmin > 10) & (pmin < 20)):
    #             print code,pmin
    # except:
    #     print '\n'
plt.legend()
plt.subplots_adjust(hspace=0.1)
plt.show()
