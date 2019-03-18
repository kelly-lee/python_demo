# -*- coding: utf-8 -*-
import tushare as ts
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import Indicators as ind
import MySQLdb as db
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import Indicators as ind


# 保存沪深复权因子
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


def save_a_daily_all(trade_date):
    """
    保存一天的股价信息
    :param trade_date:
    :return:
    """
    ts.set_token('4a988cfe3f2411b967592bde8d6e0ecbee9e364b693b505934401ea7')
    pro = ts.pro_api()
    engine = create_engine('mysql://root:root@127.0.0.1:3306/Stock?charset=utf8')
    # try:
    h_data = pro.daily(trade_date=trade_date)
    h_data = h_data[['trade_date', 'high', 'low', 'open', 'close', 'vol', 'ts_code']]
    h_data.rename(
        columns={'vol': 'volume', 'ts_code': 'symbol', 'trade_date': 'date'}, inplace=True)
    h_data['adj_close'] = 0
    # print h_data.head(5)
    h_data.to_sql('a_daily', engine, if_exists='append', index=False)
    print 'loaded'
    # except:
    #     print 'loaded error'


def save_a_daily_data(start_symbol, end_symbol, start_date, end_date):
    """
    保存指定代码和时间段的股价信息
    :param start_symbol:
    :param end_symbol:
    :param start_date:
    :param end_date:
    :return:
    """
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
            h_data = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            h_data = h_data[['trade_date', 'high', 'low', 'open', 'close', 'vol', 'ts_code']]
            h_data.rename(
                columns={'vol': 'volume', 'ts_code': 'symbol', 'trade_date': 'date'}, inplace=True)
            h_data['adj_close'] = 0
            # print h_data.head(5)
            h_data.to_sql('a_daily', engine, if_exists='append', index=False)
            print i, ts_code, name, 'loaded'
        except:
            error_code.append(ts_code)
            print i, ts_code, name, 'load error'
    print 'error code list : ', error_code


def save_a_daily_data_ind(start_date, end_date):
    """
    保存指标信息
    :param start_date:
    :param end_date:
    :return:
    """
    engine = create_engine('mysql://root:root@127.0.0.1:3306/Stock?charset=utf8')
    symbols = get_a_stock_list('a_daily')
    for symbol in symbols['symbol'].tolist():
        data = get_a_daily_data_ind(table='a_daily', symbol=symbol, start_date=start_date,
                                    end_date=end_date, append_ind=True)
        data = data[['symbol', 'date', 'close', 'pdi', 'willr', 'willr_89', 'willr_34', 'bias']]
        data.to_sql('a_daily_ind', engine, if_exists='append', index=False)
        print symbol, 'loaded'


def get_a_stock_list(table):
    """
    获得库中存在的股票列表
    :param table:
    :return:
    """
    con = db.connect('localhost', 'root', 'root', 'stock')
    sql = "SELECT distinct(symbol) FROM " + table + " where 1=1 "
    return pd.read_sql(sql, con=con)


def get_a_daily_data_ind(table='', symbol='', trade_date='', start_date='', end_date='', append_ind=False):
    """
    获得股价信息
    :param table:
    :param symbol:
    :param trade_date:
    :param start_date:
    :param end_date:
    :param append_ind:
    :return:
    """
    con = db.connect('localhost', 'root', 'root', 'stock')
    df = pd.DataFrame()
    sql = "SELECT * FROM " + table + " where 1=1 "
    if (len(symbol) > 0) & (not symbol.isspace()):
        sql += "and symbol = %(symbol)s "
    if (len(trade_date) > 0) & (not trade_date.isspace()):
        sql += "and date = %(date)s "
    if (len(start_date) > 0) & (not start_date.isspace()):
        sql += "and date >= %(start_date)s "
    if (len(end_date) > 0) & (not end_date.isspace()):
        sql += "and date <= %(end_date)s "
    sql += "order by symbol asc , date asc "
    print sql
    data = pd.read_sql(sql, params={'symbol': symbol, 'date': trade_date, 'start_date': start_date,
                                    'end_date': end_date}, con=con)
    print 'load data'
    if append_ind:
        open, close, high, low, volume = data['open'], data['close'], data['high'], data['low'], data['volume']
        ochl2ind = ind.ochl2ind(open, close, high, low, volume)
        data = data.join(ochl2ind, how='left')
    df = df.append(data)
    con.close()
    return df


# 获得沪深列表
def get_basic_stock(start='', end=''):
    ts.set_token('4a988cfe3f2411b967592bde8d6e0ecbee9e364b693b505934401ea7')
    pro = ts.pro_api()
    stock_basics = pro.stock_basic(fields='ts_code,symbol,name')
    stock_basics = stock_basics[(stock_basics.symbol < end) & (stock_basics.symbol > start)][
        ['ts_code', 'name']]
    return stock_basics


def query_by_sql(sql):
    con = db.connect('localhost', 'root', 'root', 'stock')
    return pd.read_sql(sql, con=con)


def get_a_daily_data_ind_all():
    con = db.connect('localhost', 'root', 'root', 'stock')
    sql = """
    select * from a_daily_ind 
    """
    df = pd.read_sql(sql, con=con)
    df['date'] = df['date'].astype(str)
    df.index = df['date']
    df.drop(columns=['date'], inplace=True)
    row = 7
    col = 4
    fig = plt.figure(figsize=(16, 16))
    i = 1
    for date in df.index[-20:]:
        print date
        ax = fig.add_subplot(row, col, i)
        i = i + 1
        # print df.loc['2019-02-' + i, 'willr'].describe()
        # data = df.groupby(by='date').mean()
        # print data.head(5)
        # ax = plt.axes()
        w = df.loc[date, 'willr']
        w = w.sort_values()
        w.index = np.arange(1, len(w) + 1)
        ax.bar(w.index, w)
        # ax.plot(data['willr'])
        # ax.plot(data['willr_89'])
        # ax = plt.twinx()
        # ax.plot(data['bias'], c='green')
    plt.show()


def get_buy():
    fig = plt.figure(figsize=(16, 6))
    sql = """
            select * from a_daily_ind  where willr_89<-10  and date = '20190314' order by willr_89 limit 0,20
        """
    df = query_by_sql(sql)
    print df['symbol'].tolist()
    row = 4
    col = 5
    show(row, col, df['symbol'].tolist())


def show(row, col, symbols):
    fig = plt.figure(figsize=(16, 8))

    i = 1
    for symbol in symbols:
        data = get_a_daily_data_ind(table='a_daily_ind', symbol=symbol, trade_date='', start_date='2018-10-01',
                                    end_date='2019-03-18', append_ind=False)
        data['pct'] = data['close'].pct_change()
        print data.tail()
        close, willr, willr_34, willr_89, = data['close'], data['willr'], data['willr_34'], data['willr_89']
        bias, pdi = data['bias'], data['pdi']

        ax = fig.add_subplot(row, col, i)
        # ax.plot(bias, c='grey')
        ax.plot(close, c='grey')
        buy = close[
            # ind.LESS_THAN(bias.shift(1), -3) & ind.BOTTOM(bias) &
            ind.LESS_THAN(willr.shift(1), -88)
            # & ind.BOTTOM(willr)
            & ind.LESS_THAN(willr_34.shift(1), -88)
            # & ind.BOTTOM(willr_34)
            # & ind.LESS_THAN(willr_89.shift(1), -88)
            # & ind.BOTTOM(willr_89)
            ]
        # print buy
        ax.scatter(buy.index, buy, s=10, c='green')
        ax.set_ylabel(symbol)
        ax = plt.twinx()
        # ax.plot(bias)

        # ax.plot(willr)
        # ax.plot(willr_34)
        # ax.plot(willr_89)

        i = i + 1

    plt.legend()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.3, wspace=0.3)
    plt.show()


def get_symbols(sql):
    return query_by_sql(sql)['symbol'].tolist()


def pct():
    sql = """
          select distinct(symbol) from a_daily order by symbol asc
          """
    symbols = query_by_sql(sql)
    symbols = symbols['symbol'].tolist()
    aa = pd.DataFrame()
    i = 0
    for symbol in symbols:
        # if (symbol != '300216.SZ'):
        #     continue
        sql = "select * from a_daily where symbol = '" + symbol + "' order by date asc"
        df = query_by_sql(sql)

        df['pct'] = df['close'].pct_change() * 100
        # print df
        # print df.sort_values(['pct'], ascending=False).head(20)
        c5 = len(df[df['pct'] > 5])
        c6 = len(df[df['pct'] > 6])
        c7 = len(df[df['pct'] > 7])
        c8 = len(df[df['pct'] > 8])
        c9 = len(df[df['pct'] > 9])
        print symbol, c8, c9
        aa = aa.append(pd.DataFrame([[symbol, c5, c6, c7, c8, c9]]))
    aa.to_csv("aa.cvs")
    print aa


def high():
    sql = """
          select distinct(symbol) from a_daily_ind order by symbol asc
          """
    symbols = query_by_sql(sql)
    symbols = symbols['symbol'].tolist()
    bb = pd.DataFrame()
    i = 0
    for symbol in symbols:
        sql = "select * from a_daily_ind where symbol = '" + symbol + "' order by date asc"
        df = query_by_sql(sql)

        df['pct'] = df['close'].pct_change() * 100
        print df[df['pct'].shift(-1) > 9]

        bb = bb.append(df[df['pct'].shift(-1) > 9])
    bb.to_csv("bb.cvs")
    print bb


def hot():
    aa = pd.read_csv('aa.cvs')
    aa = aa[aa['1'] > 30]
    symbols = aa['0'].tolist()
    row = 6
    col = 4
    show(row, col, symbols[0:row * col])


if __name__ == '__main__':
    bb = pd.read_csv('bb.cvs')

    print bb.describe()[['willr', 'willr_34', 'willr_89', 'bias']]
    bb = bb.sort_values(['bias'], ascending=True)
    # plt.scatter(bb.index, bb['willr'])
    # bb.dropna
    # print bb['willr']
    cc = bb['bias'].dropna()
    plt.hist(cc, bins=50)
    plt.show()
    # high()
    # row = 6
    # col = 4
    # size = row * col
    # symbols = get_symbols("select distinct(symbol) from a_daily_ind where willr <-88  and date = '2019-03-18' order by bias limit 0," + str(size))
    # show(row, col, symbols[0:size])
    # aa = pd.read_csv('aa.cvs')
    # print len(aa), len(aa[aa['5'] > 0]), len(aa[aa['5'] > 5]), len(aa[aa['5'] > 10]), len(aa[aa['5'] > 15]), len(
    #     aa[aa['5'] > 20])
    # print len(aa), len(aa[aa['1'] > 0]), len(aa[aa['1'] > 5]), len(aa[aa['1'] > 10]), len(aa[aa['1'] > 15]), len(
    #     aa[aa['1'] > 20])
    # print aa.describe().T
    # print aa.sort_values(['5'], ascending=False)
    # aa = aa[aa['1'] > 30]
    # print aa['0'].tolist()
    #
    # symbols = aa['0'].tolist()
    # print len(symbols)
    # row = 6
    # col = 4
    # show(row, col, symbols[0:row * col])

    # pct()
    # df = df[df['pct'] > 5]
    # print df.count()
    # df = df.sort_values(['symbol'], ascending=True, inplace=False)
    # print df
    # pc = df.groupby(['symbol']).size()
    # print pc.sort_values(ascending=False, inplace=False)

    # print df.info()
    # print df[['symbol','close','pct']]
    # abc = df.groupby(['symbol', 'pct']).size().unstack()
    # print abc

    # get_buy()
    # get_a_daily_data_ind_all()
    # save_a_daily_all(trade_date='20190318')
    # save_a_daily_data_ind(start_date='2018-10-01', end_date='2019-03-18')
