# -*- coding: utf-8 -*-
import tushare as ts
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import matplotlib.pyplot as plt
import TushareStore as store
import Indicators as ind
import mpl_finance as mpf
import pandas_datareader.data as web


def get_chart_data_from_web(code, start_date='', end_date='', append_ind=True):
    data = web.DataReader(code, data_source='yahoo', start=start_date, end=end_date)
    data.rename(columns={'Open': 'open', 'Close': 'close', 'High': 'high', 'Low': 'low', 'Volume': 'volume'},
                inplace=True)
    if append_ind:
        open, close, high, low, volume = data['open'], data['close'], data['high'], data['low'], data['volume']
        ochl2ind = ind.ochl2ind(open, close, high, low, volume)
        data = data.join(ochl2ind, how='left')
    data['date'] = data.index
    data['date'].apply(lambda date: date.date())
    data.index = np.arange(0, 0 + len(data))
    return data


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
    for period in periods:
        if data.__contains__('willr'):
            willr = data['willr']
        else:
            close, high, low = data['close'], data['high'], data['low']
            willr = ind.WILLR(high, low, close, time_period=period)
        ax.plot(willr, label='wr%d' % period)
    drawHline(ax, hlines)


def drawDMI(ax, data, periods=[6, 6], hlines=[10, 12, 16, 20, 22]):
    if data.__contains__('pdi') & data.__contains__('mdi') & data.__contains__('adx') & data.__contains__('adxr'):
        pdi = data['pdi']
        mdi = data['mdi']
        adx = data['adx']
        adxr = data['adxr']
    else:
        close, high, low = data['close'], data['high'], data['low']
        pdi, mdi = ind.DI(high, low, close, time_period=periods[0])
        adx = ind.ADX(high, low, close, time_period=periods[1])
        adxr = ind.ADXR(high, low, close, time_period=periods[1])
    ax.plot(pdi, label='pdi%d' % periods[0])
    # ax.plot(mdi, label='mdi%d' % periods[0])
    # ax.plot(adx, label='adx%d' % periods[1])
    # ax.plot(adxr, label='adxr%d' % periods[1])
    drawHline(ax, hlines)


def drawCCI(ax, data, periods=[14], hlines=[-231, -138, -110, -83, 50]):
    for period in periods:
        if data.__contains__('cci'):
            cci = data['cci']
        else:
            close, high, low = data['close'], data['high'], data['low']
            cci = ind.CCI(high, low, close, time_period=period)
        ax.plot(cci, label='cci%d' % period)
    drawHline(ax, hlines)


################################################################################################

def drawMin(ax, data):
    close, high, low = data['close'], data['high'], data['low']
    # pdi, mdi = ind.DI(high, low, close, time_period=6)
    # ax.plot(pdi)
    pdi2, mdi2 = ind.DI(high, low, close, time_period=14)
    ax.plot(pdi2, c='orange')
    drawHline(ax, [10, 12, 16, 20, 22])

    # ax = plt.twinx()
    # ax.plot(close, label='willr', c='grey')
    # cross = close[(pdi2 < 12)]
    # ax.scatter(cross.index, cross, c='red')

    # pdi = data['pdi']

    # ax.plot(pdi, label='pdi', c='orange')
    # drawHline(ax, [10, 12])
    willr = data['willr']
    ax = plt.twinx()
    ax.plot(willr, label='willr', c='green')
    # drawHline(ax, [-98, -83.5])


def drawInd(ind='', ax=None, data=None):
    if ind == 'K':
        drawK(ax, data)
    if ind == 'SMA':
        drawSMA(ax, data)
    if ind == 'BBANDS':
        drawBBANDS(ax, data)
    if ind == 'WR':
        drawWR(ax, data)
    if ind == 'DMI':
        drawDMI(ax, data)
    if ind == 'KDJ':
        drawKDJ(ax, data)
    if ind == 'CCI':
        drawCCI(ax, data)


def drawAll(data, types=[['K', 'SMA']]):
    row = len(types)
    fig = plt.figure(figsize=(16, 4 * row))
    i = 0
    for type in types:
        i += 1
        ax = fig.add_subplot(row, 1, i)
        j = 0
        for ind in type:
            j += 1
            if j == 2:
                ax = plt.twinx()
            drawInd(ind, ax, data)
            drawDate(ax, data)
    plt.legend()
    plt.subplots_adjust(hspace=0.1)
    # plt.grid(True)
    plt.show()


def drawBuy(codes):
    col = 2
    matrix = np.reshape(codes, (-1, col))
    row = len(matrix)
    fig = plt.figure(figsize=(8 * col, 4 * row))
    i = 0
    for code in codes:
        i += 1
        data = store.get_chart_data_from_db("000%d.SZ" % code, '20180101')
        # data = get_chart_data_from_web(code, '1/1/2018', '1/30/2019')
        close, pdi, wr = data['close'], data['pdi'], data['willr']
        ax = fig.add_subplot(row, col, i)
        ax.plot(close, c='grey')
        # buy = close[(wr <= -98)]
        # ax.scatter(buy.index, buy, c='red')
        # buy = close[(wr <= -93) & (wr > -98)]
        # ax.scatter(buy.index, buy, c='orange')
        # buy = close[(wr <= -88) & (wr > -93)]
        # ax.scatter(buy.index, buy, c='yellow')
        # buy = close[(wr <= -83) & (wr > -88)]
        # ax.scatter(buy.index, buy, c='green')

        buy = close[(pdi <= 10) & (wr < -88)]
        ax.scatter(buy.index, buy, c='red')
        buy = close[(pdi <= 12) & (pdi > 10) & (wr < -88)]
        ax.scatter(buy.index, buy, c='orange')
        buy = close[(pdi <= 16) & (pdi > 12) & (wr < -88)]
        ax.scatter(buy.index, buy, c='yellow')
        buy = close[(pdi <= 20) & (pdi > 16) & (wr < -88)]
        ax.scatter(buy.index, buy, c='green')

    plt.legend()
    plt.subplots_adjust(hspace=0.1)
    plt.show()


# data = get_chart_data_from_web('AMZN', '1/1/2018', '1/30/2019')
# data = store.get_chart_data_from_db('000001.SZ', '20180101')
# types = [['K', 'SMA'], ['WR', 'DMI'], ['KDJ'], ['CCI']]
# drawAll(data, types=types)
################################################
a = np.arange(0, 12)
# 501 516 598 589 582 586 589
print np.reshape(a, (-1, 4)).shape[0]
codes = ['AMZN', 'AAPL', 'GOOG', 'FB']
codes = np.random.randint(600, 700, [10])
print codes
drawBuy(codes)
