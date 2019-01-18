# -*- coding: utf-8 -*-

# from __future__ import print_function

# Author: Gael Varoquaux gael.varoquaux@normalesup.org
# License: BSD 3 clause
import tushare as ts
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import TushareStore as store
import pandas as pd
import pandas_datareader.data as web
from sklearn import cluster, covariance, manifold

from matplotlib.font_manager import FontProperties

print(__doc__)
import MySQLdb as db
from sqlalchemy import create_engine
import TushareStore as store

error_code = []
engine = create_engine('mysql://root:root@127.0.0.1:3306/Stock?charset=utf8')
companys = store.get_usa_company(sector='Finance')
for index, company in companys.iterrows():
    id = company['id']
    if id < 10462:
        continue
    symbol = company['Symbol']
    print company['id'], symbol
    try:
        nasdaq_daily = web.DataReader(symbol, start='1/1/2015', data_source='yahoo')
        nasdaq_daily.index = nasdaq_daily.index.to_period("D")
        nasdaq_daily['symbol'] = symbol
        nasdaq_daily.rename(columns={'Open': 'open', 'Close': 'close', 'High': 'high', 'Low': 'low', 'Volume': 'volume',
                                     'Adj Close': 'adj_close'}, inplace=True)
        nasdaq_daily.to_sql('usa_finance_daily', engine, if_exists='append')
    except:
        print company['id'], symbol, 'load error'
        error_code.append(company['id'])
print error_code


