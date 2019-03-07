# -*- coding: utf-8 -*-
import tushare as ts
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import Indicators as ind
import MySQLdb as db
import pandas_datareader.data as web


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


def get_a_stock_list(table):
    con = db.connect('localhost', 'root', 'root', 'stock')
    df = pd.DataFrame()
    sql = "SELECT distinct(symbol) FROM " + table + " where 1=1 "
    return pd.read_sql(sql, con=con)


def save_a_daily_data_ind(start_date, end_date):
    engine = create_engine('mysql://root:root@127.0.0.1:3306/Stock?charset=utf8')
    symbols = get_a_stock_list('a_daily')
    for symbol in symbols['symbol'].tolist():
        data = get_a_daily_data_ind(table='a_daily', symbol=symbol, start_date=start_date,
                                    end_date=end_date, append_ind=True)
        data = data[['symbol', 'date', 'close', 'pdi', 'willr', 'willr_89', 'bias']]
        data.to_sql('a_daily_ind', engine, if_exists='append', index=False)
        print symbol, 'loaded'


def get_a_daily_data_ind(table='', symbol='', trade_date='', start_date='', end_date='', append_ind=False):
    con = db.connect('localhost', 'root', 'root', 'stock')
    df = pd.DataFrame()
    sql = "SELECT symbol,date,open,close,adj_close,high,low,volume FROM " + table + " where 1=1 "
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


# 获得沪深日线
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


# 保存美股公司信息
def save_usa_company():
    engine = create_engine('mysql://root:root@127.0.0.1:3306/Stock?charset=utf8')
    company = pd.DataFrame()
    nasdaq_company = pd.read_csv('CompanyList_NASDAQ.data')

    nasdaq_company['Exchange'] = 'NASDAQ'
    company = company.append(nasdaq_company)
    nasdaq_company.drop(['Unnamed: 9'], axis=1, inplace=True)
    nasdaq_company.to_sql('usa_company', engine, if_exists='append')

    nyse_company = pd.read_csv('CompanyList_NYSE.data')
    nyse_company['Exchange'] = 'NYSE'
    nyse_company.drop(['Unnamed: 8'], axis=1, inplace=True)
    nyse_company.to_sql('usa_company', engine, if_exists='append')
    company = company.append(nyse_company)

    amex_company = pd.read_csv('CompanyList_AMEX.data')
    amex_company['Exchange'] = 'AMEX'
    amex_company.drop(['Unnamed: 8'], axis=1, inplace=True)
    amex_company.to_sql('usa_company', engine, if_exists='append')
    company = company.append(amex_company)


# 查询美股公司信息
# Health Care 卫生保健|Finance 金融|Consumer Services 消费服务|Technology 技术|Miscellaneous 杂|Capital Goods 资本货物
# Energy 能源|Public Utilities 公共设施|Basic Industries 基础工业|Transportation 运输|Consumer Non-Durables 非耐用消费品|Consumer Durables 耐用消费品
# NASDAQ纳斯达克,NYSE纽约证券交易所,AMEX美国证券交易所
def get_usa_company(exchange='', sector='', symbol=''):
    con = db.connect('localhost', 'root', 'root', 'stock')
    engine = create_engine('mysql://root:root@127.0.0.1:3306/Stock?charset=utf8')
    sql = "SELECT * FROM usa_company_4 where 1=1  and Name_CN is not null "
    if (len(exchange) > 0) & (not exchange.isspace()):
        sql += "and exchange = %(exchange)s "
    if (len(sector) > 0) & (not sector.isspace()):
        sql += "and sector = %(sector)s "
    if (len(symbol) > 0) & (not symbol.isspace()):
        sql += "and symbol = %(symbol)s "
    sql += "order by id asc, exchange asc,sector asc,symbol asc "
    print sql
    company = pd.read_sql(sql, params={'exchange': exchange, 'sector': sector, 'symbol': symbol}, con=con)
    con.close()
    return company


# 查询美股日行情
def get_usa_daily_data_ind(sector='', symbol='', trade_date='', start_date='', end_date='', append_ind=False):
    con = db.connect('localhost', 'root', 'root', 'stock')
    df = pd.DataFrame()
    # if (len(symbol) != 0) | (not symbol.isspace()):
    #     sector = get_usa_company(symbol=symbol)['Sector'].values[0]
    #     print sector
    # sql = "SELECT symbol,date,open,close,adj_close,high,low,volume FROM usa_%s_daily where 1=1 " % sector.lower()
    sql = "SELECT symbol,date,open,close,adj_close,high,low,volume FROM usa_technology_daily where 1=1 "
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
    if append_ind:
        open, close, high, low, volume = data['open'], data['close'], data['high'], data['low'], data['volume']
        ochl2ind = ind.ochl2ind(open, close, high, low, volume)
        data = data.join(ochl2ind, how='left')
    df = df.append(data)
    con.close()
    return df


def save_usa_daily_data(sector, company_id, symbol, start='1/1/2015'):
    error_code = []
    engine = create_engine('mysql://root:root@127.0.0.1:3306/Stock?charset=utf8')
    companys = get_usa_company(sector=sector)
    for index, company in companys.iterrows():
        id = company['id']
        if id < company_id:
            continue
        symbol = company['Symbol']
        print company['id'], symbol
        try:
            nasdaq_daily = web.DataReader(symbol, start='1/1/2015', data_source='yahoo')
            nasdaq_daily.index = nasdaq_daily.index.to_period("D")
            nasdaq_daily['symbol'] = symbol
            nasdaq_daily.rename(
                columns={'Open': 'open', 'Close': 'close', 'High': 'high', 'Low': 'low', 'Volume': 'volume',
                         'Adj Close': 'adj_close'}, inplace=True)
            nasdaq_daily.to_sql('usa_finance_daily', engine, if_exists='append')
        except:
            print company['id'], symbol, 'load error'
            error_code.append(company['id'])
    print error_code


def save_usa_daily_data(tabel, symbol, start, end):
    engine = create_engine('mysql://root:root@127.0.0.1:3306/Stock?charset=utf8')
    try:
        nasdaq_daily = web.DataReader(symbol, start=start, end=end, data_source='yahoo')
        nasdaq_daily.index = nasdaq_daily.index.to_period("D")
        nasdaq_daily['symbol'] = symbol
        nasdaq_daily.rename(
            columns={'Open': 'open', 'Close': 'close', 'High': 'high', 'Low': 'low', 'Volume': 'volume',
                     'Adj Close': 'adj_close'}, inplace=True)
        nasdaq_daily.to_sql(tabel, engine, if_exists='append')
    except:
        print  symbol, 'save error'


def query_by_sql(sql):
    con = db.connect('localhost', 'root', 'root', 'stock')
    return pd.read_sql(sql, con=con)


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


import matplotlib.pyplot as plt


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


if __name__ == '__main__':
    # get_a_daily_data_ind_all()

    # save_a_daily_all(trade_date='20190307')
    save_a_daily_data_ind(start_date='2018-10-01', end_date='2019-03-07')

    # date high,low,open,close,volume,adj_close,id,symbol
    #
    # trade_date(date),high,low,open,close,vol(volume),close,id,symbol
    # a_daliy_300
    #     print get_a_stock_list('a_daily')
    #     print get_a_daily_data_ind(table='a_daily', start_date='20180101', end_date='20190125', append_ind=True)

    #     save_a_daily_data_ind(start_date='2018-10-01', end_date='2019-02-27')
    #     save_a_daily_data(start_symbol='000', end_symbol='700', start_date='20190225', end_date='20190226')
    #     save_usa_company()
    #     df = get_usa_company(symbol='ASFI')
    #     print df[['Symbol', 'Exchange', 'Sector']]
    # df = get_usa_daily_data_ind(symbol=df['Symbol'].values[0])
    # print df
    # print df.groupby(['Sector'])['id'].count()
    #
    # df = get_usa_company()
    # print df.groupby(['Sector'])['id'].count()
