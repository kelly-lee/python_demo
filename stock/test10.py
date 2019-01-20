# -*- coding: utf-8 -*-
# from __future__ import print_function

# Author: Gael Varoquaux gael.varoquaux@normalesup.org
# License: BSD 3 clause

import sys
import pandas_datareader.data as web
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import pandas as pd

from sklearn import cluster, covariance, manifold
from bs4 import BeautifulSoup
import urllib2
from bs4 import UnicodeDammit
import re
import TushareStore as store

import sys

import MySQLdb as db


def save_name_cn():
    html = open('CompanyList_Xueqiu_Usa_Tech.html', 'r').read()
    soup = BeautifulSoup(html, features="lxml")
    a = soup.select("td a")
    df = pd.DataFrame()
    i = 0
    for link in a:
        row = i / 3
        col = i % 3
        if i % 3 < 2:
            df.at[row, col] = link.text
        i += 1
    df.rename(columns={0: 'symbol', 1: 'name_cn'}, inplace=True)
    df.to_csv('CompanyList_Xueqiu_Usa_Tech.csv', index=False, header=True, encoding='utf8')


def merge_company():
    company = pd.DataFrame()
    amex = pd.read_csv('CompanyList_AMEX.csv')
    amex.drop(["LastSale", "Summary Quote"], inplace=True, axis=1)
    amex.rename(
        columns={"Symbol": "symbol", "Name": "name", "MarketCap": "market_cap", "IPOyear": "ipo_year",
                 "Sector": "sector",
                 "Industry": "industry"}, inplace=True)
    amex.drop(amex.columns[- 1], inplace=True, axis=1)
    nasdaq = pd.read_csv('CompanyList_NASDAQ.csv')
    nasdaq.drop(["LastSale", "Summary Quote", "ADR TSO"], inplace=True, axis=1)
    nasdaq.drop(nasdaq.columns[- 1], inplace=True, axis=1)
    nasdaq.rename(
        columns={"Symbol": "symbol", "Name": "name", "MarketCap": "market_cap", "IPOyear": "ipo_year",
                 "Sector": "sector",
                 "Industry": "industry"}, inplace=True)
    nyse = pd.read_csv('CompanyList_NYSE.csv')
    nyse.drop(["LastSale", "Summary Quote"], inplace=True, axis=1)
    nyse.rename(
        columns={"Symbol": "symbol", "Name": "name", "MarketCap": "market_cap", "IPOyear": "ipo_year",
                 "Sector": "sector",
                 "Industry": "industry"}, inplace=True)
    nyse.drop(nyse.columns[- 1], inplace=True, axis=1)
    company = company.append(amex)
    company = company.append(nasdaq)
    company = company.append(nyse)
    company.to_csv('CompanyList.csv', index=False, header=True, encoding='utf-8')


nasdaq = web.DataReader('^IXIC', start='1/1/2015', data_source='yahoo')
print nasdaq

sql_max = """
select t1.symbol as symbol,t1.adj_close as max_close,unix_timestamp(t1.date) as max_ts from nasdaq_technology_daily as t1
inner join (select symbol ,max(adj_close) as max_adj_close from nasdaq_technology_daily where adj_close >1 group by symbol  )t2
on t1.symbol = t2.symbol and t1.adj_close = t2.max_adj_close
"""

sql_min = """
select t1.symbol as symbol,t1.adj_close as min_close,unix_timestamp(t1.date) as min_ts from nasdaq_technology_daily as t1
inner join (select symbol ,min(adj_close) as min_adj_close from nasdaq_technology_daily where adj_close >1  group by symbol  )t2
on t1.symbol = t2.symbol and t1.adj_close = t2.min_adj_close
"""

con = db.connect('localhost', 'root', 'root', 'stock')
pf_max = pd.read_sql(sql_max, con=con)
print pf_max[pf_max.symbol == 'YI']
pf_max = pf_max.drop_duplicates(['symbol'])
pf_min = pd.read_sql(sql_min, con=con)
print pf_min[pf_min.symbol == 'YI']
pf_min = pf_min.drop_duplicates(['symbol'])
df = pd.merge(pf_max, pf_min, how='inner', on=['symbol'])
df['drt'] = df.max_ts - df.min_ts
df['roc'] = (df.max_close - df.min_close) / df.min_close
# df = df.drop_duplicates(['symbol'])
df.sort_values(by=['roc'], ascending=[0], inplace=True)
df = df[df.drt > 0]
print len(df)

save_name_cn()
cn = pd.read_csv("CompanyList_Xueqiu_Usa_Tech.csv")
cn = pd.merge(cn, df, how='outer', on=['symbol'])
print len(cn)
cn.sort_values(by=['roc'], ascending=[0], inplace=True)
cn.to_csv('test.csv', index=False, header=True, encoding='utf8')

stocks = df[df.max_ts < df.min_ts]
print len(stocks)
print stocks
col = 10
row = 12
fig = plt.figure(figsize=(16, 8))
i = 0
df = df.iloc[0:120]
print df
for symbol in df['symbol'].tolist():
    i += 1
    prices = store.get_usa_daily_data_ind(symbol=symbol)
    prices.index = prices.date
    # print prices
    ax = fig.add_subplot(row, col, i)
    ax.set_ylabel(symbol)
    ax.plot(prices['adj_close'])
    ax.set_xticks([])
    ax.set_yticks([])
    ax = ax.twinx()
    ax.plot(nasdaq['Adj Close'], color='grey')
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()
