# -*- coding: utf-8 -*-
import tushare as ts
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import Indicators as ind
import MySQLdb as db


# 保存复权因子
def save_adj_factor(start_symbol='000', end_symbol='601', trade_date=''):
    ts.set_token('4a988cfe3f2411b967592bde8d6e0ecbee9e364b693b505934401ea7')
    pro = ts.pro_api()
    engine = create_engine('mysql://root:root@127.0.0.1:3306/Stock?charset=utf8')
    stock_basics = pro.stock_basic(fields='ts_code,symbol,name')
    for index, row in stock_basics.iterrows():
        ts_code = row['ts_code']
        symbol = row['symbol']
        if symbol > end_symbol:
            continue
        if symbol < start_symbol:
            continue
        df = pro.adj_factor(ts_code=ts_code, trade_date=trade_date)
        adj_factor = df.groupby('adj_factor').min()
        adj_factor.to_sql('adj_factor', engine, if_exists='append')
        print 'adj', ts_code


# 保存日线
def save_daily_data(start_symbol='000', end_symbol='601', trade_date=''):
    ts.set_token('4a988cfe3f2411b967592bde8d6e0ecbee9e364b693b505934401ea7')
    pro = ts.pro_api()
    engine = create_engine('mysql://root:root@127.0.0.1:3306/Stock?charset=utf8')
    stock_basics = pro.stock_basic(fields='ts_code,symbol,name')
    i = 0
    error_code = []
    for index, row in stock_basics.iterrows():
        ts_code = row['ts_code']
        name = row['name']
        symbol = row['symbol']
        i += 1
        if symbol > end_symbol:
            continue
        if symbol < start_symbol:
            continue
        try:
            h_data = pro.daily(ts_code=ts_code, trade_date=trade_date)
            h_data.to_sql('daily_data', engine, if_exists='append')
            print i, ts_code, name, 'loaded'
        except:
            error_code.append(ts_code)
            print i, ts_code, name, 'load error'
    print 'error code list : ', error_code


def get_basic_stock(start='', end=''):
    ts.set_token('4a988cfe3f2411b967592bde8d6e0ecbee9e364b693b505934401ea7')
    pro = ts.pro_api()
    stock_basics = pro.stock_basic(fields='ts_code,symbol,name')
    stock_basics = stock_basics[(stock_basics.symbol < end) & (stock_basics.symbol > start)][
        ['ts_code', 'name']]
    return stock_basics


def get_daily_data(type='300', size=0, start_date='', end_date=''):
    engine = create_engine('mysql://root:root@127.0.0.1:3306/Stock?charset=utf8')
    df = pd.read_sql('daily_data_%s' % type, engine)
    df.drop(['id', 'index'], axis=1, inplace=True)
    print 'load data'
    inds = pd.DataFrame()
    i = 0
    for ts_code in df['ts_code'].drop_duplicates():
        i = i + 1
        if (size != 0) & (i > size):
            break
        stock = df[(df.ts_code == ts_code) & (df.trade_date >= start_date) & (df.trade_date <= end_date)]
        stock = stock.sort_values(by=['trade_date'], ascending=True)
        high, low, open, close, volume = stock['high'], stock['low'], stock['open'], stock['close'], stock['vol']
        ochl2ind = ind.ochl2ind(open, close, high, low, volume)
        stock.join(ochl2ind, how='left').to_sql('daily_ind_%s' % type, engine, if_exists='append')
        # 计算指标
        inds = inds.append(ochl2ind)
        print 'ind', ts_code
    df = df.join(inds, how='left').dropna()
    return df


def get_daily_data_ind(ts_code='', trade_date='', start_date='', end_date='', append_ind=False):
    con = db.connect('localhost', 'root', 'root', 'stock')
    if (len(ts_code) > 0) & (not ts_code.isspace()):
        table_suffixs = [ts_code[0:3]]
    else:
        table_suffixs = ['000', '002', '300', '600']
    df = pd.DataFrame()
    for table_suffix in table_suffixs:
        sql = "SELECT ts_code,trade_date,open,close,high,low,vol as volume FROM daily_data_%s where 1=1 " % table_suffix
        if (len(ts_code) > 0) & (not ts_code.isspace()):
            sql += "and ts_code = %(ts_code)s "
        if (len(trade_date) > 0) & (not trade_date.isspace()):
            sql += "and trade_date = %(trade_date)s "
        if (len(start_date) > 0) & (not start_date.isspace()):
            sql += "and trade_date >= %(start_date)s "
        if (len(end_date) > 0) & (not end_date.isspace()):
            sql += "and trade_date >= %(end_date)s "
        sql += "order by trade_date asc "
        print sql
        data = pd.read_sql(sql, params={'ts_code': ts_code, 'trade_date': trade_date, 'start_date': start_date,
                                        'end_date': end_date}, con=con)
        if append_ind:
            open, close, high, low, volume = data['open'], data['close'], data['high'], data['low'], data['volume']
            ochl2ind = ind.ochl2ind(open, close, high, low, volume)
            data = data.join(ochl2ind, how='left')
        df = df.append(data)
    con.close()
    return df


def get_chart_data_from_db(code='', start_date='', end_date='', append_ind=True):
    data = get_daily_data_ind(ts_code=code, start_date=start_date, end_date=end_date, append_ind=append_ind)
    data.drop(['ts_code'], axis=1, inplace=True)
    data = data.dropna()
    data.rename(columns={'trade_date': 'date'}, inplace=True)
    data.index = np.arange(0, 0 + len(data))
    return data


def save_nasdaq_company():
    nasdaq_company = pd.read_csv('NASDAQ_companylist.csv')
    engine = create_engine('mysql://root:root@127.0.0.1:3306/Stock?charset=utf8')
    nasdaq_company.to_sql('nasdaq_company', engine, if_exists='append')


def get_nasdaq_company():
    engine = create_engine('mysql://root:root@127.0.0.1:3306/Stock?charset=utf8')
    nasdaq_company = pd.read_sql('nasdaq_company', engine)
    return nasdaq_company


import pandas_datareader.data as web


nasdaq_daily = web.DataReader('GOOG', start='1/1/2018', data_source='yahoo')
print nasdaq_daily


engine = create_engine('mysql://root:root@127.0.0.1:3306/Stock?charset=utf8')
nasdaq_companys = get_nasdaq_company()
for index, nasdaq_company in nasdaq_companys.iterrows():
    if nasdaq_company['index'] < 7:
        continue
    symbol = nasdaq_company['Symbol']
    print nasdaq_company['index'], symbol
    nasdaq_daily = web.DataReader(symbol, start='1/1/2018', data_source='yahoo')
    nasdaq_daily.index = nasdaq_daily.index.to_period("D")
    nasdaq_daily['symbol'] = symbol
    nasdaq_daily.rename(columns={'Open': 'open', 'Close': 'close', 'High': 'high', 'Low': 'low', 'Volume': 'volume',
                                 'Adj Close': 'adj_close'}, inplace=True)
    nasdaq_daily.to_sql('nasdaq_daily', engine, if_exists='append')
