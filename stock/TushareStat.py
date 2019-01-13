# -*- coding: utf-8 -*-
import tushare as ts
from sqlalchemy import create_engine
import pandas as pd
import stock.Indicators as ind
import matplotlib.pyplot as plt
import numpy as np
import TushareStore as ts

# pdi 16 | 5.8 | 0~10~12~16~20~24~50
# pdi  24|8|1,18.5,23.7,29,99

# slow_j 7.4|17.3| -63.5~-3~6.5~17~97.5
# slow_j 77|25|-17|59|80|95|158

# willr -88|11 |-100~-97~-92~-83.5~-20
# willr  -32|23|-98,-52,-25,-11,-0


# roc  -9.2 | 7.7 | -70~-12~-7.5~-4~0
# cci -109 | 41 | -231~-138~-110~-83~50
# cci 57|69|-124,-4,68,112,226


df = ts.get_daily_data('300', size=0, start_date='20180101', end_date='20190131')

# print df[(df['trade_date'] == '20190111') & (df['pdi'] < 16)]

# inds = ['roc', 'pdi', 'slow_j', 'willr', 'cci']
inds = ['vol_roc', 'macd_hist', 'mfi', 'aroon_up', 'trix']
size = len(inds)
fig = plt.figure(figsize=(12, 6))

for i in range(size):
    ind_min = df[(df['close'] == df['min'])][inds[i]]
    print ind_min.describe()
    ax = fig.add_subplot(2, 5, i + 1)
    ax.scatter(ind_min, np.full((ind_min.count()), ind_min.index), label=inds[i], s=0.1, c='r', alpha=0.5)
    # ax.title = inds[i]
for i in range(size):
    ind_max = df[(df['close'] == df['max'])][inds[i]]
    print ind_max.describe()
    ax = fig.add_subplot(2, 5, i + 6)
    ax.scatter(ind_max, np.full((ind_max.count()), ind_max.index), label=inds[i], s=0.1, c='g', alpha=0.5)
    # ax.title = inds[i]
plt.legend()
plt.subplots_adjust(hspace=0.1)
plt.show()
