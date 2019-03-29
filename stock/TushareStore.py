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


def get_buy(sql, row, col):
    size = row * col
    sql = sql + " limit 0," + str(size)
    show(row, col, get_symbols(sql))


# 根据股票代码显示股价
def show(row, col, symbols):
    fig = plt.figure(figsize=(16, 8))

    i = 1
    for symbol in symbols:
        data = get_a_daily_data_ind(table='a_daily_ind', symbol=symbol, trade_date='', start_date='2019-01-01',
                                    end_date='2019-03-30', append_ind=False)
        data['pct'] = data['close'].pct_change()
        print data.tail()
        close, willr, willr_34, willr_89, = data['close'], data['willr'], data['willr_34'], data['willr_89']
        bias, pdi = data['bias'], data['pdi']

        ax = fig.add_subplot(row, col, i)
        # ax.plot(bias, c='grey')
        buy = close[
            ind.LESS_THAN(willr, -88) &
            ind.LESS_THAN(willr_34, -88)
            # ind.GREAT_THAN(bias, 3)
            # & ind.GREAT_THAN(willr_89, -28)
            # & ind.LESS_THAN(willr, -70)

            # ind.LESS_THAN(bias.shift(1), -3) & ind.BOTTOM(bias) &
            # & ind.GREAT_THAN(willr, -40)
            # & ind.LESS_THAN(bias.shift(), 3)
            # & ind.BOTTOM(willr)
            # & ind.LESS_THAN(willr_34.shift(1), -88)
            # & ind.BOTTOM(willr_34)
            # & ind.LESS_THAN(willr_89.shift(1), -88)
            # & ind.BOTTOM(willr_89)
            ]
        ax.scatter(buy.index, buy, s=20, c='green')
        ax.plot(close, c='grey')
        # ax.plot(ind.MAX(close, 10), c='grey')
        # ax.plot(ind.MIN(close, 10), c='grey')
        ax.set_xticks([])
        ax.set_yticks([])

        # print buy

        ax.set_ylabel(symbol)
        ax = plt.twinx()
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.plot(bias)

        # ax.plot(willr)
        # ax.plot(willr_34)
        # ax.plot(willr_89)

        i = i + 1

    plt.legend()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.3, wspace=0.3)
    plt.show()


# 根据sql查询符合条件的股票代码
def get_symbols(sql=''):
    if (len(sql) == 0) | (sql.isspace()):
        sql = "select distinct(symbol) from a_daily order by symbol asc"
    return query_by_sql(sql)['symbol'].tolist()


# 统计配置股票涨幅频率
def pct():
    symbols = get_symbols()
    data = pd.DataFrame()
    # i = 0
    for symbol in symbols:
        # i = i + 1
        # if i > 5:
        #     break
        sql = "select * from a_daily where symbol = '" + symbol + "' order by date asc"
        df = query_by_sql(sql)
        df['pct'] = df['close'].pct_change() * 100
        p5 = len(df[df['pct'] > 5])
        p6 = len(df[df['pct'] > 6])
        p7 = len(df[df['pct'] > 7])
        p8 = len(df[df['pct'] > 8])
        p9 = len(df[df['pct'] > 9])
        s5 = len(df[df['pct'] < -5])
        s6 = len(df[df['pct'] < -6])
        s7 = len(df[df['pct'] < -7])
        s8 = len(df[df['pct'] < -8])
        s9 = len(df[df['pct'] < -9])
        print symbol
        p8 = p8 - p9
        p7 = p7 - p8
        p6 = p6 - p7
        p5 = p5 - p6
        s8 = s8 - s9
        s7 = s7 - s8
        s6 = s6 - s7
        s5 = s5 - s6
        data = data.append(pd.DataFrame([[symbol, p5, p6, p7, p8, p9, s5, s6, s7, s8, s9]],
                                        columns=['symbol', 'p5', 'p6', 'p7', 'p8', 'p9', 's5', 's6', 's7', 's8', 's9']))
        data.to_csv("pct_data.cvs")


def high():
    symbols = get_symbols()
    data = pd.DataFrame()
    for symbol in symbols:
        sql = "select * from a_daily_ind where symbol = '" + symbol + "' order by date asc"
        df = query_by_sql(sql)
        df['pct'] = df['close'].pct_change() * 100
        data = data.append(df[df['pct'].shift(-1) > 9])
    data.to_csv("high_data.cvs")


def low():
    symbols = get_symbols()
    data = pd.DataFrame()
    for symbol in symbols:
        sql = "select * from a_daily_ind where symbol = '" + symbol + "' order by date asc"
        df = query_by_sql(sql)
        df['pct'] = df['close'].pct_change() * 100
        data = data.append(df[df['pct'].shift(-1) < -9])
    data.to_csv("low_data.cvs")


def hot():
    aa = pd.read_csv('aa.cvs')
    aa = aa[aa['1'] > 30]
    symbols = aa['0'].tolist()
    row = 6
    col = 4
    show(row, col, symbols[0:row * col])


# bias -3,3~19,29
# willr_89 -36 -28  -2 0

if __name__ == '__main__':
    get_buy("select distinct(symbol) from a_daily_ind where willr<-88 and willr_34<-88 and date = '2019-03-25'", 5, 5)
    # pct()
    # save_a_daily_all(trade_date='20190327')
    # save_a_daily_data_ind(start_date='2018-10-01', end_date='2019-03-27')
# low()
# get_buy()
# datas = []
# # datas.append(query_by_sql("select * from a_daily_ind"))
# datas.append(pd.read_csv('bb.cvs'))
# datas.append(pd.read_csv('cc.cvs'))
# colors = ['red', 'blue', 'yellow']
# i = 0
# for data in datas:
#     data = data.sort_values(['willr_89'], ascending=True)
#     data = data['willr_89'].dropna()
#     plt.hist(data, facecolor=colors[i], bins=50)
#     i = i + 1
#     # print data.describe()[['willr', 'willr_34', 'willr_89', 'bias']]
# plt.show()

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
