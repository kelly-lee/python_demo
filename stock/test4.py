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
nasdaq_companys = store.get_nasdaq_company()
nasdaq_companys = nasdaq_companys[nasdaq_companys['Sector'] == 'Technology']
for index, nasdaq_company in nasdaq_companys.iterrows():
    # if nasdaq_company['index'] < 83:
    #     continue
    symbol = nasdaq_company['Symbol']
    print nasdaq_company['index'], symbol
    try:
        nasdaq_daily = web.DataReader(symbol, start='1/1/2015', data_source='yahoo')
        nasdaq_daily.index = nasdaq_daily.index.to_period("D")
        nasdaq_daily['symbol'] = symbol
        nasdaq_daily.rename(columns={'Open': 'open', 'Close': 'close', 'High': 'high', 'Low': 'low', 'Volume': 'volume',
                                     'Adj Close': 'adj_close'}, inplace=True)
        nasdaq_daily.to_sql('nasdaq_technology_daily', engine, if_exists='append')
    except:
        print nasdaq_company['index'], symbol, 'load error'
        error_code.append(nasdaq_company['index'])
print error_code
