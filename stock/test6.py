# -*- coding: utf-8 -*-
import tushare as ts
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import matplotlib.pyplot as plt

import Indicators as ind
import mpl_finance as mpf
import pandas_datareader.data as web
import talib
import Charts

import MySQLdb as db


# 83~100
def get_nasdaq_daily_data_ind(symbol='', trade_date='', start_date='', end_date='', append_ind=False):
    con = db.connect('localhost', 'root', 'root', 'stock')
    df = pd.DataFrame()
    sql = "SELECT symbol,date,open,close,adj_close,high,low,volume FROM nasdaq_daily where 1=1 "
    if (len(symbol) > 0) & (not symbol.isspace()):
        sql += "and symbol = %(symbol)s "
    if (len(trade_date) > 0) & (not trade_date.isspace()):
        sql += "and date = %(date)s "
    if (len(start_date) > 0) & (not start_date.isspace()):
        sql += "and date >= %(start_date)s "
    if (len(end_date) > 0) & (not end_date.isspace()):
        sql += "and date <= %(end_date)s "
    sql += "order by date asc "
    print sql
    data = pd.read_sql(sql, params={'symbol': symbol, 'date': trade_date, 'start_date': start_date,
                                    'end_date': end_date}, con=con)
    if append_ind:
        open, close, high, low, volume = data['open'], data['close'], data['high'], data['low'], data['volume']
        ochl2ind = ind.ochl2ind(open, close, high, low, volume)
        data = data.join(ochl2ind, how='left')
    df = df.append(data)
    con.close()
    return df


data = get_nasdaq_daily_data_ind('GOOG', start_date='2016-01-01', end_date='2018-12-31')
types = [['C', 'MINMAX'], ['C', 'WR'], ['C', 'DMI'], ['C', 'KDJ'], ['C', 'CCI'], ['C', 'RSI'], ['C', 'MACD']]

types = [['C', 'MINMAX'], ['C', 'EMV'], ['C', 'TRIX'], ['C', 'OBV'], ['C', 'MFI'], ['C', 'RSI'], ['C', 'ROC']]
Charts.drawAll('GOOG', data, types=types)
