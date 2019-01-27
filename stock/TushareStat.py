# -*- coding: utf-8 -*-
import tushare as ts
from sqlalchemy import create_engine
import pandas as pd
import stock.Indicators as ind
import matplotlib.pyplot as plt
import numpy as np
import UsaStore as store
import scipy.stats as st


def normfun(x):
    mean = x.mean()
    sigma = x.std()
    pdf = np.exp(-((x - mean) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    return pdf


def drawOne(data, ax):
    # ax.scatter(ind_min, np.full((ind_min.count()), ind_min.index), label=inds[i], s=0.1, c='r', alpha=0.5)
    # ax.title = inds[i]
    print data.describe()
    mean = data.mean()
    quantiles = data.quantile(q=[0.25, 0.5, 0.75])
    x = data.sort_values().values
    y = st.norm.pdf(x, x.mean(), x.std())
    ax.axvline(x=mean, color='grey', linestyle="--", linewidth=1)
    for quantile in quantiles:
        ax.axvline(x=quantile, color='grey', linestyle="--", linewidth=1)
    ax.hist(x, bins=50, alpha=0.4)
    ax.plot(x, y)
    ax.set_xticks(quantiles)
    ax.tick_params(axis='x', rotation=60)
    ax.legend(fontsize=9, ncol=3)


inds = ['bias', 'pdi', 'slow_j', 'willr', 'cci', 'mdi', 'adx', 'adxr', 'vol_roc', 'macd_hist']
# inds = ['vol_roc', 'macd_hist', 'mfi', 'aroon_up', 'trix']

sector = 'public_utilities'
start_date = '2018-01-01'
end_date = '2019-01-21'
usa_company_list = store.get_usa_company_list(exchange='', sector=sector)

df = pd.DataFrame()
for symbol in usa_company_list['symbol'].tolist():
    data = store.get_usa_daily_data_ind(sector=sector, symbol=symbol, start_date=start_date, end_date=end_date,
                                        append_ind=True)
    df = df.append(data)
df.dropna(inplace=True)
size = len(inds)
fig = plt.figure(figsize=(12, 6))

for i in range(size):
    ind_min = df[(df['close'] == df['min_20'])][inds[i]]
    ax = fig.add_subplot(2, 10, i + 1)
    drawOne(ind_min, ax)

for i in range(size):
    ind_max = df[(df['close'] == df['max_20'])][inds[i]]
    ax = fig.add_subplot(2, 10, i + 11)
    drawOne(ind_max, ax)

plt.setp(plt.gca().get_xticklabels(), rotation=60)
plt.legend()
plt.subplots_adjust(hspace=0.5)
plt.show()
