# -*- coding: utf-8 -*-

import sys

reload(sys)
sys.setdefaultencoding('utf8')
import time
import tushare as ts
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import Indicators as ind
import MySQLdb as db
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import Indicators as ind
from sklearn import tree
from sklearn.model_selection import train_test_split
import xgboost as xgb
from matplotlib.font_manager import FontProperties
import Charts as charts


def get_ts_pro():
    ts.set_token('4a988cfe3f2411b967592bde8d6e0ecbee9e364b693b505934401ea7')
    pro = ts.pro_api()
    return pro


# 获得每天行情
def get_ts_daily(trade_date):
    pro = get_ts_pro()
    data = pro.daily(trade_date=trade_date)
    data = data[['trade_date', 'high', 'low', 'open', 'close', 'vol', 'ts_code', 'pct_chg']]
    data.rename(
        columns={'vol': 'volume', 'ts_code': 'symbol', 'trade_date': 'date'}, inplace=True)
    return data


# 获得股票基本信息
def get_ts_basic():
    pro = get_ts_pro()
    df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry,list_date,area')
    return df


def save_stock_basic():
    engine = create_engine('mysql://root:root@127.0.0.1:3306/Stock?charset=utf8')
    data = get_ts_basic()
    data.to_sql('stock_basic', engine, if_exists='append', index=False)


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


# 保存一天的股价信息
def save_a_daily_all(trade_date):
    h_data = get_ts_daily(trade_date=trade_date)
    h_data = h_data[['symbol', 'date', 'high', 'low', 'open', 'close', 'volume']]
    h_data['adj_close'] = 0
    engine = create_engine('mysql://root:root@127.0.0.1:3306/Stock?charset=utf8')
    try:
        h_data.to_sql('a_daily', engine, if_exists='append', index=False)
        print 'loaded'
    except:
        print 'loaded error'


# 保存每天行情
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


# 保存每天指标信息
def save_a_daily_data_ind(start_date, end_date):
    engine = create_engine('mysql://root:root@127.0.0.1:3306/Stock?charset=utf8')
    symbols = query_symbols()
    for symbol in symbols:
        data = query_a_daily_data(symbol, trade_date='', start_date=start_date, end_date=end_date)
        open, close, high, low, volume = data['open'], data['close'], data['high'], data['low'], data['volume']
        ochl2ind = ind.ochl2buy(open, close, high, low, volume)
        data = data.join(ochl2ind, how='left')
        data.to_sql('a_daily_ind', engine, if_exists='append', index=False)
        print symbol, 'loaded'


# 统计大涨大跌次数和窗口期内累积最大涨幅和最大跌幅
def save_industry_sat():
    industry_sats = pd.DataFrame()
    basic_stock = query_basic_stock()
    print basic_stock.head(5)
    i = 0
    for industry in basic_stock['industry'].unique():
        i = i + 1
        symbols = basic_stock[basic_stock['industry'] == industry]['ts_code']
        industry_sat = pd.DataFrame()
        for symbol in symbols:
            print symbol
            # try:
            data = query_a_daily_data_ind(symbol=symbol, trade_date='', start_date='2019-01-01',
                                          end_date='2019-04-30')
            sat = ind.sat(data)
            sat['symbol'] = symbol
            industry_sat = industry_sat.append(sat)
            industry_sat = industry_sat.sort_values(by=['pct_sum_90_max'], ascending=False)
            # except:
            #     print 'error', symbol
            # print industry_top
        industry_sats = industry_sats.append(industry_sat)
    basic_stock['symbol'] = basic_stock['ts_code']
    industry_sats = pd.merge(basic_stock, industry_sats, on=['symbol'], how='inner')
    industry_sats.to_csv('industry_sats.csv')
    engine = create_engine('mysql://root:root@127.0.0.1:3306/Stock?charset=utf8')
    # industry_sats.to_sql('industry_sats', engine, if_exists='append', index=False)

    industry_sats_1 = industry_sats[0:1000]
    industry_sats_1.to_sql('industry_sats', engine, if_exists='append', index=False)
    industry_sats_2 = industry_sats[1000:2000]
    industry_sats_2.to_sql('industry_sats', engine, if_exists='append', index=False)
    industry_sats_3 = industry_sats[2000:3000]
    industry_sats_3.to_sql('industry_sats', engine, if_exists='append', index=False)
    industry_sats_4 = industry_sats[3000:]
    industry_sats_4.to_sql('industry_sats', engine, if_exists='append', index=False)


def query_by_sql(sql='', params={}):
    con = db.connect('localhost', 'root', 'root', 'stock', charset='utf8')
    data = pd.read_sql(sql=sql, params=params, con=con)
    con.close()
    return data


# 根据sql查询符合条件的股票代码
def query_symbols(sql=''):
    if (len(sql) == 0) | (sql.isspace()):
        sql = "select distinct(symbol) from a_daily order by symbol asc"
    return query_by_sql(sql)['symbol'].tolist()


def query_a_daily_data(symbol='', trade_date='', start_date='', end_date=''):
    sql = "SELECT * FROM a_daily where 1=1 "
    if (len(symbol) > 0) & (not symbol.isspace()):
        sql += "and symbol = %(symbol)s "
    if (len(trade_date) > 0) & (not trade_date.isspace()):
        sql += "and date = %(date)s "
    if (len(start_date) > 0) & (not start_date.isspace()):
        sql += "and date >= %(start_date)s "
    if (len(end_date) > 0) & (not end_date.isspace()):
        sql += "and date <= %(end_date)s "
    sql += "order by symbol asc , date asc "
    params = {'symbol': symbol, 'date': trade_date, 'start_date': start_date, 'end_date': end_date}
    return query_by_sql(sql, params)


def query_a_daily_data_ind(symbol='', trade_date='', start_date='', end_date=''):
    sql = "SELECT * FROM a_daily_ind where 1=1 "
    if (len(symbol) > 0) & (not symbol.isspace()):
        sql += "and symbol = %(symbol)s "
    if (len(trade_date) > 0) & (not trade_date.isspace()):
        sql += "and date = %(date)s "
    if (len(start_date) > 0) & (not start_date.isspace()):
        sql += "and date >= %(start_date)s "
    if (len(end_date) > 0) & (not end_date.isspace()):
        sql += "and date <= %(end_date)s "
    sql += "order by symbol asc , date asc "
    params = {'symbol': symbol, 'date': trade_date, 'start_date': start_date, 'end_date': end_date}
    print sql, params
    return query_by_sql(sql, params)


def query_basic_stock():
    return query_by_sql(sql="select * from stock_basic")


def draw_buy(sql, row, col):
    size = row * col
    sql = sql + " limit 0," + str(size)
    show(row, col, query_symbols(sql))


# 画涨幅累加对比图（多股重叠）
def draw_pct_sum(symbols, names, start_date, end_date):
    # symbols = ['601128.SH', '601577.SH', '002936.SZ',
    #            '603323.SH', '002839.SZ', '002807.SZ',
    #            '000001.SZ', '002142.SZ', '600036.SH','601009.SH',
    #            '601166.SH', '601997.SH', '601998.SH','600908.SH','002948.SZ']
    font = FontProperties(fname='/System/Library/Fonts/PingFang.ttc', size=10)
    fig = plt.figure(figsize=(8, 8))
    i = -1
    legend_names = []
    for symbol in symbols:
        i = i + 1
        data = query_a_daily_data_ind(symbol=symbol, trade_date='', start_date=start_date,
                                      end_date=end_date)
        data['pct_sum'] = data['pct'].cumsum()
        if (data['pct_sum'].max() <= 50):
            continue
        # if (data['pct_sum'].max() > 50):
        #     continue
        legend_names.append(names.iloc[i])
        print symbol, data['pct_sum'].max()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(data['pct_sum'])
        ax.set_yticks(np.arange(0, 80, 10))

        ax.legend(labels=legend_names, loc=2, prop=font)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.3, wspace=0.3)
    plt.show()


# 画涨幅对比图
def draw_pct():
    pp5_list = []
    ps5_list = []
    dates = pd.date_range('1/1/2019', '4/30/2019')
    for date in dates:
        date = date.strftime('%Y%m%d')
        data = get_ts_daily(date)
        size = len(data)
        if size == 0:
            continue
        pp5 = len(data[data['pct_chg'] >= 5]) * 1.0 / size
        ps5 = len(data[data['pct_chg'] <= -5]) * 1.0 / size
        pp5_list.append(pp5)
        ps5_list.append(ps5)
        print date, pp5, ps5

    plt.plot(pp5_list)
    plt.plot(ps5_list)
    plt.show()


def test_draw_pct_sum():
    basic = get_ts_basic()
    basic = basic[basic['industry'] == '生物制药']
    symbols = basic['ts_code']
    names = basic['name']
    draw_pct_sum(symbols=symbols, names=names, start_date='2019-01-01', end_date='2019-04-30')


# 画 willr整体分布图
def draw_willr_bar():
    data = query_by_sql(sql="select date,willr from a_daily_ind order by date asc")
    row = 5
    col = 4
    fig = plt.figure(figsize=(8, 8))
    i = 1
    for date in data['date'].unique()[-20:]:
        willr = data[data['date'] == date]['willr']
        willr = willr.sort_values()
        willr.index = np.arange(1, len(willr) + 1)
        ax = fig.add_subplot(row, col, i)
        ax.bar(willr.index, willr)
        ax.set_ylabel(date)
        i = i + 1
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.3, wspace=0.3)
    plt.show()


###################################################


# 根据股票代码显示股价
def show(row, col, symbols, names, start_date='2019-01-01', end_date='2019-04-30'):
    font = FontProperties(fname='/System/Library/Fonts/PingFang.ttc', size=10)
    fig = plt.figure(figsize=(24, 16))
    i = 0
    for symbol in symbols:
        data = query_a_daily_data_ind(symbol=symbol, trade_date='', start_date=start_date,
                                      end_date=end_date)
        ax = fig.add_subplot(row, col, i / col * 2 * col + i % col + 1)
        # ax.plot(data['close'])
        charts.drawK(ax, data)
        charts.drawSMA(ax, data, periods=[3, 20, 60])
        charts.drawDate(ax, data)
        ax.set_ylabel(names[i], fontproperties=font)
        show_buy(ax, data, 'k')
        show_trend(ax, data)
        ax = fig.add_subplot(row, col, i / col * 2 * col + i % col + col + 1)
        charts.drawMACD(ax, data)
        # charts.drawDate(ax, data)

        ax.plot(data['macd'], label='diff')
        ax.plot(data['macd_signal'], label='dea')
        ax.axhline(y=0, color='grey', linestyle="--", linewidth=1)
        show_buy(ax, data, 'macd')
        show_trend(ax, data)

        # ax.set_ylabel(symbol)
        # ax = plt.twinx()
        # ax.set_xticks([])
        # ax.set_yticks([])
        i = i + 1
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.1, wspace=0.1)
    plt.show()


def get_daily_choose():
    a_daily = query_by_sql("SELECT * FROM Stock.a_daily_ind where  willr<-80  and date='2019-04-01' order by willr asc")
    a_daily['ts_code'] = a_daily['symbol']
    industry_top = pd.read_csv("industry_tops_1.csv", index_col=0)
    industry_top['symbol'] = industry_top.index
    print industry_top.head()
    m = pd.merge(industry_top, a_daily, on=['symbol'], how='inner')
    m.to_csv('daily.csv')


def show_buy(ax, data, type):
    close, low, high = data['close'], data['low'], data['high']
    ma3, ma5, ma20, ma60 = data['sma_3'], data['sma_5'], data['sma_20'], data['sma_60']
    diff, dea = data['macd'], data['macd_signal']
    close, willr, willr_34, willr_89, = data['close'], data['willr'], data['willr_34'], data['willr_89']
    bias, pdi = data['bias'], data['pdi']
    buy = ind.LESS_THAN(willr, -88)
    # & ind.LESS_THAN(willr_34, -88)
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

    buy = ind.TOUCH(close, ma20) & ind.TOUCH(diff, dea, 0.3)

    # diff,dea 死叉 close>ma20 会上涨 close<ma20 会下跌
    # 上升趋势 diff,dea,close,ma20 同时死叉 最佳买点 会涨
    # 下跌趋势 diff,dea,close,ma20 同时金叉 反弹最高点，会跌

    # 上升趋势 diff,dea,close,ma20 同时金叉 最佳买点 会涨
    buy = ind.DEAD_CROSS(diff, dea)

    # diff,dea 金叉，close>ma20 会震荡回调 close<ma20 会补涨
    # 上升趋势 diff,dea,close,ma20 同时金叉 最佳买点 会涨
    # 下跌趋势 diff,dea,close,ma20 同时金叉 反弹最高点，会跌
    # buy = ind.GOLDEN_CROSS(diff, dea) | ind.DEAD_CROSS(diff, dea)

    if type == 'k':
        ax.scatter(close[buy].index, close[buy], s=20, c='black')
    else:
        ax.scatter(diff[buy].index, diff[buy], s=20, c='black')


def show_trend(ax, data):
    close, low, high = data['close'], data['low'], data['high']
    ma3, ma5, ma20, ma60 = data['sma_3'], data['sma_5'], data['sma_20'], data['sma_60']
    diff, dea = data['macd'], data['macd_signal']

    ax = plt.twinx()
    ax.set_yticks([])
    # 强上涨
    ax.bar(data[ind.UP(ma3) & (0 < dea) & (dea < diff)].index, 1, facecolor='darkred', width=1,
           alpha=0.5)
    # 强下跌
    ax.bar(data[ind.DOWN(ma3) & (diff < dea) & (dea < 0)].index, 1, facecolor='darkgreen', width=1,
           alpha=0.5)
    # 弱下跌
    ax.bar(data[ind.DOWN(ma3) & (0 < diff) & (diff < dea)].index, 1, facecolor='yellowgreen', width=1,
           alpha=0.5)
    # 弱上涨
    ax.bar(data[ind.UP(ma3) & (dea < diff) & (diff < 0)].index, 1, facecolor='pink',
           width=1, alpha=0.5)
    # # 努力上升
    # ax.bar(data[ind.BETWEEN(ma60, low, high) & ind.UP(ma3) & (dea < diff) & (diff < 0)].index, 1, facecolor='pink',
    #        width=1, alpha=0.7)
    # # 努力下跌
    # ax.bar(data[ind.BETWEEN(ma60, low, high) & ind.DOWN(ma3) & (diff < 0) & (0 < dea)].index, 1, facecolor='lightgreen',
    #        width=1, alpha=0.7)

    # ax.bar(data[ind.DOWN(ma20) & ind.UP(dea)].index, 1, facecolor='grey', width=1,
    #        alpha=0.7)
    # ax.bar(data[ind.UP(ma20) & ind.DOWN(dea)].index, 1, facecolor='black', width=1,
    #        alpha=0.7)

    # ax.bar(data[ind.BETWEEN(close, ma20 - close * 0.01, ma20 + close * 0.01) & ind.BETWEEN(diff, (dea - 0.08), (dea + 0.08))].index, 1,
    #        facecolor='black', width=1, alpha=0.7)

    #
    #
    # # 零上跌超底背离 努力下跌+弱下跌
    # ax.bar(data[(close < ma60) & (ma60 < ma20) & (0 < diff) & (diff > dea)].index, 1, facecolor='grey', width=1,
    #        alpha=0.7)
    # # 零下涨超顶背离  努力上升+弱上升
    # ax.bar(data[(ma20 < ma60) & (ma60 < close) & (diff < dea) & (dea < 0)].index, 1, facecolor='grey', width=1,
    #        alpha=0.7)
    # # 跨零轴超前底背离 努力下跌+ 强下跌
    # ax.bar(data[(close < ma60) & (ma60 < ma20) & (diff < dea) & (dea < 0)].index, 1, facecolor='grey', width=1,
    #        alpha=0.7)
    # # 跨零轴超前顶背离  努力上升+强上升
    # ax.bar(data[(ma20 < ma60) & (ma60 < close) & (0 < dea) & (dea < diff)].index, 1, facecolor='rosybrown', width=1,
    #        alpha=0.7)
    # # 顶部反背离  强下跌 弱上涨
    # ax.bar(data[(close < ma20) & (ma20 < ma60) & (dea < diff) & (diff < 0)].index, 1, facecolor='grey', width=1,
    #        alpha=0.7)
    # # 底部反背离  弱下跌 弱下跌
    # ax.bar(data[(ma60 < close) & (close < ma20) & (0 < diff) & (diff < dea)].index, 1, facecolor='grey', width=1,
    #        alpha=0.7)


def d():
    data = query_a_daily_data_ind(symbol='000568.SZ', trade_date='', start_date='2018-01-01',
                                  end_date='2019-04-30')
    close, low, high, willr, willr_34, willr_89, = data['close'], data['low'], data['high'], data['willr'], data[
        'willr_34'], data['willr_89']
    bias, pdi = data['bias'], data['pdi']
    ma3, ma20, ma60 = close, data['sma_20'], data['sma_60']
    diff, dea = data['macd'], data['macd_signal']

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(2, 1, 1)

    print data
    ax.plot(close)
    # plt.plot(ma3)
    ax.plot(ma20)
    ax.plot(ma60)
    # dd(ax, data, 'k')

    ax = fig.add_subplot(2, 1, 2)
    ax.plot(diff, label='diff')
    ax.plot(dea, label='dea')
    ax.axhline(y=0, color='grey', linestyle="--", linewidth=1)
    ax.set_ylabel('MACD')
    # dd(ax, data, 'diff')
    plt.show()


if __name__ == '__main__':
    # d()
    # 保存股票基本信息
    # save_stock_basic()
    # 保存每天行情
    # save_a_daily_all(trade_date='20190417')
    # 保存所有买卖技术指标
    save_a_daily_data_ind(start_date='2000-01-01', end_date='2019-04-17')
    # 统计大涨大跌次数和窗口期内累积最大涨幅和最大跌幅
    # save_industry_sat()

    # 画某时段涨幅图
    # test_draw_pct_sum()
    # print query_basic_stock()
    # 根据sql显示K线图面板
    # sql = """
    # select symbol,name
    # from a_daily_ind
    # inner join stock_basic  on a_daily_ind.symbol = stock_basic.ts_code
    # where date = '2019-04-15' order by  pct_sum_3 asc limit 1800,24
    # """
    # stocks = query_by_sql(sql)
    # print stocks
    # show(8, 6, stocks['symbol'].tolist(), stocks['name'].tolist())
    # willr整体分布图
    # draw_willr_bar()
