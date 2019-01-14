# -*- coding: utf-8 -*-
import tushare as ts
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import matplotlib.pyplot as plt
import MySQLdb as db
import Indicators as ind
import mpl_finance as mpf
import pandas_datareader.data as web


def getDailyDate(ts_code='', trade_date='', start_date='', end_date='', append_ind=False):
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


def drawHline(ax, hlines):
    for hline in hlines:
        ax.axhline(y=hline, color='grey', linestyle="--", linewidth=1)


def drawDate(ax, data):
    # weeksLoc = mpl.dates.WeekdayLocator()
    # ax.xaxis.set_minor_locator(weeksLoc)
    date = data['date']
    date_index = np.arange(0, 0 + len(date))
    ax.set_xticks(date_index)
    ax.set_xticklabels(date)
    for label in ax.get_xticklabels():
        label.set_visible(False)
    for label in ax.get_xticklabels()[::50]:
        label.set_visible(True)


def drawK(ax, data):
    date, open, close, high, low, volume = data.index, data['open'], data['close'], data['high'], data['low'], data[
        'volume']
    date_index = np.arange(0, 0 + len(date))
    mpf.candlestick_ohlc(ax, zip(date_index, open, high, low, close),
                         colorup='red', colordown='green')


def drawBBANDS(ax, data, period=20):
    if data.__contains__('upper_band') & data.__contains__('middle_band') & data.__contains__('lower_band'):
        upper_band = data['upper_band']
        middle_band = data['middle_band']
        lower_band = data['lower_band']
    else:
        close = data['close']
        upper_band, middle_band, lower_band = ind.BBANDS(close, time_period=period)
    ax.plot(upper_band, label='upper_band')
    ax.plot(middle_band, label='middle_band')
    ax.plot(lower_band, label='lower_band')


def drawSMA(ax, data, periods=[5, 10, 20, 30, 60, 120]):
    for period in periods:
        if data.__contains__('sma%d' % period):
            sma = data['sma%d' % period]
        else:
            close = data['close']
            sma = ind.SMA(close, period)
        ax.plot(sma, label='sma%d' % period, linewidth=1)


def drawEMA(ax, data, periods=[5, 10, 20, 30, 60, 120]):
    for period in periods:
        if data.__contains__('ema%d' % period):
            ema = data['ema%d' % period]
        else:
            close = data['close']
            ema = ind.SMA(close, period)
        ax.plot(ema, label='ema%d' % period)


################################################################################################
def drawKDJ(ax, data, periods=[9, 3, 3], hlines=[-14, -3, 6.5, 17, 95]):
    if data.__contains__('slow_k') & data.__contains__('slow_d'):
        slow_k = data['slow_k']
        slow_d = data['slow_d']
    else:
        close, high, low = data['close'], data['high'], data['low']
        slow_k, slow_d = ind.STOCH(high, low, close, fastk_period=periods[0], slowk_period=periods[1],
                                   slowd_period=periods[2])
    ax.plot(slow_d, label='d')
    ax.plot(slow_k, label='k')
    ax.plot(3 * slow_k - 2 * slow_d, label='j')
    drawHline(ax, hlines)
    # ax.fill_between(time, 80, rsi, where=rsi >= 80, facecolor='green')
    # ax.fill_between(time, 20, rsi, where=rsi <= 20, facecolor='red')


def drawWR(ax, data, periods=[6], hlines=[-98, -93, -88, -83.5, -25, -11]):
    close, high, low = data['close'], data['high'], data['low']
    for period in periods:
        wr = ind.WILLR(high, low, close, time_period=period)
        ax.plot(wr, label='wr%d' % period)
    drawHline(ax, hlines)


def drawDMI(ax, prices, periods=[14, 6], hlines=[10, 12, 16, 20, 22]):
    high, low, close = prices[0], prices[1], prices[2]
    pdi, mdi = ind.DI(high, low, close, time_period=periods[0])
    adx = ind.ADX(high, low, close, time_period=periods[1])
    adxr = ind.ADXR(high, low, close, time_period=periods[1])
    ax.plot(pdi, label='pdi%d' % periods[0])
    # ax.plot(mdi, label='mdi%d' % periods[0])
    # ax.plot(adx, label='adx%d' % periods[1])
    # ax.plot(adxr, label='adxr%d' % periods[1])
    # ax.bar(pdi.index, (pdi - mdi).clip_lower(0), facecolor='r')
    # ax.bar(pdi.index, (pdi - mdi).clip_upper(0), facecolor='g')
    drawHline(ax, hlines)


def drawCCI(ax, prices, periods=[14], hlines=[-231, -138, -110, -83, 50]):
    high, low, close = prices[0], prices[1], prices[2]
    for period in periods:
        cci = ind.CCI(high, low, close, time_period=period)
        ax.plot(cci, label='cci%d' % period)
    drawHline(ax, hlines)


# 连接mysql，获取连接的对象
df = getDailyDate(ts_code='000001.SZ', start_date='20180101', append_ind=False)
df.drop(['ts_code'], axis=1, inplace=True)
df = df.dropna()
df.rename(columns={'trade_date': 'date'}, inplace=True)
df.index = np.arange(0, 0 + len(df))
# df = web.DataReader('AAPL', data_source='yahoo', start='1/1/2018', end='1/30/2019')
# df.rename(columns={'Open': 'open', 'Close': 'close', 'High': 'high', 'Low': 'low', 'Volume': 'volume'}, inplace=True)
# df['date'] = df.index
# df['date'].apply(lambda date: date.date())
# df.index = np.arange(0, 0 + len(df))
# print df
# print df.info()
################################################
fig = plt.figure(figsize=(16, 8))
ind_size = 2
ax = fig.add_subplot(ind_size, 1, 1)
drawK(ax, df)
# drawSMA(ax, df)
drawEMA(ax, df)
# drawBBANDS(ax, df)
drawDate(ax, df)
ax = fig.add_subplot(ind_size, 1, 2)
drawKDJ(ax, df)
drawDate(ax, df)
plt.legend()
plt.subplots_adjust(hspace=0.1)
# plt.grid(True)
plt.show()
