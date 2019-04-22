# -*- coding: utf-8 -*-
import tushare as ts
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import matplotlib.pyplot as plt
import mpl_finance as mpf
import pandas_datareader.data as web
import stock.TushareStore as TushareStore
import stock.Indicators as ind


# import stock.UsaStore as store

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


def drawC(ax, data):
    close = data['close']
    ax.plot(close, color='grey')


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
        if data.__contains__('sma_%d' % period):
            sma = data['sma_%d' % period]
        elif data.__contains__('sma'):
            sma = data['sma']
        else:
            close = data['close']
            sma = ind.SMA(close, period)
        ax.plot(sma, label='sma_%d' % period, linewidth=1)


def drawEMA(ax, data, periods=[5, 10, 20, 30, 60, 120]):
    for period in periods:
        if data.__contains__('ema%d' % period):
            ema = data['ema%d' % period]
        elif data.__contains__('ema'):
            ema = data['ema']
        else:
            close = data['close']
            ema = ind.SMA(close, period)
        ax.plot(ema, label='ema%d' % period)


def drawMINMAX(ax, data, periods=[20]):
    for period in periods:
        if data.__contains__('min_%d' % period) & data.__contains__('max_%d' % period):
            min = data['min_%d' % period]
            max = data['max_%d' % period]
        # elif data.__contains__('min') & data.__contains__('max'):
        #     min = data['min']
        #     max = data['max']
        else:
            close = data['close']
            min = ind.MIN(close, period)
            max = ind.MAX(close, period)
        ax.plot(min, label='min%d' % period, linewidth=1)
        ax.plot(max, label='max%d' % period, linewidth=1)
        ax.set_ylabel('MIN_MAX')


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
    ax.set_ylabel('KDJ')


def drawWR(ax, data, periods=[6, 89], hlines=[-98, -83.5, -25, -11]):
    for period in periods:
        if data.__contains__('willr_%d' % period):
            willr = data['willr_%d' % period]
        elif data.__contains__('willr'):
            willr = data['willr']
        else:
            close, high, low = data['close'], data['high'], data['low']
            willr = ind.WILLR(high, low, close, time_period=period)
        ax.plot(willr, label='wr%d' % period)
    drawHline(ax, hlines)
    ax.set_ylabel('WR')


def drawDMI(ax, data, periods=[6, 14], hlines=[10, 12, 16, 20, 22]):
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
    ax.set_ylabel('DMI')


def drawCCI(ax, data, periods=[14], hlines=[-231, -138, -110, -83, 50]):
    for period in periods:
        if data.__contains__('cci'):
            cci = data['cci']
        else:
            close, high, low = data['close'], data['high'], data['low']
            cci = ind.CCI(high, low, close, time_period=period)
        ax.plot(cci, label='cci%d' % period)
    drawHline(ax, hlines)
    ax.set_ylabel('CCI')


################################################################################################

def drawRSI(ax, data, periods=[6, 12, 24], hlines=[20, 50, 80]):
    for period in periods:
        if data.__contains__('rsi%d' % period):
            rsi = data['rsi%d' % period]
        elif data.__contains__('rsi'):
            rsi = data['rsi']
        else:
            price = data['close']
            rsi = ind.RSI(price, time_period=period)
        ax.plot(rsi, label='rsi%d' % period)
    drawHline(ax, hlines)
    ax.set_ylabel('RSI')


def drawMACD(ax, data, periods=[12, 26, 9]):
    close = data['close']
    if data.__contains__('macd') & data.__contains__('macd_signal') & data.__contains__('macd_hist'):
        macd = data['macd']
        macd_signal = data['macd_signal']
        macd_histogram = data['macd_hist']
    # else:
    #     macd, macd_signal, macd_histogram = ind.MACD(close, fast_period=periods[0], slow_period=periods[1],
    #                                                  signal_period=periods[2])
    ax.plot(macd, label='macd%d' % periods[0])
    ax.plot(macd_signal, label='macd_singal%d' % periods[1])
    ax.bar(close.index, macd_histogram.clip(lower=0), facecolor='r')
    ax.bar(close.index, macd_histogram.clip(upper=0), facecolor='g')
    drawHline(ax, [0])
    # ax.set_ylabel('MACD')
    ax.set_yticks([])


def drawTRIX(ax, data, periods=[14], hlines=[]):
    for period in periods:
        if data.__contains__('trix%d' % period):
            trix = data['rsi%d' % period]
        elif data.__contains__('trix'):
            trix = data['trix']
        else:
            price = data['close']
            trix = ind.TRIX(price, time_period=period)
        ax.plot(trix, label='trix%d' % period)
    drawHline(ax, hlines)
    ax.set_ylabel('TRIX')


def drawEMV(ax, data, periods=[14], hlines=[]):
    for period in periods:
        if data.__contains__('emv%d' % period):
            emv = data['emv%d' % period]
        elif data.__contains__('emv'):
            emv = data['emv']
        else:
            low, high, volume = data['low'], data['high'], data['volume']
            emv = ind.EMV(low, high, volume, time_period=period)
        ax.plot(emv, label='emv%d' % period)
    drawHline(ax, hlines)
    ax.set_ylabel('EMV')


def drawMFI(ax, data, periods=[14], hlines=[]):
    for period in periods:
        if data.__contains__('mfi%d' % period):
            mfi = data['mfi%d' % period]
        elif data.__contains__('mfi'):
            mfi = data['mfi']
        else:
            high, low, close, volume = data['low'], data['high'], data['close'], data['volume']
            mfi = ind.MFI(high, low, close, volume, time_period=period)
        ax.plot(mfi, label='mfi%d' % period)
    drawHline(ax, hlines)
    ax.set_ylabel('MFI')


def drawOBV(ax, data, hlines=[]):
    if data.__contains__('obv'):
        obv = data['obv']
    else:
        close, volume = data['close'], data['volume']
        obv = ind.OBV(close, volume)
    ax.plot(obv, label='obv')
    drawHline(ax, hlines)
    ax.set_ylabel('OBV')


def drawROC(ax, data, periods=[14], hlines=[]):
    for period in periods:
        if data.__contains__('roc%d' % period):
            roc = data['roc%d' % period]
        elif data.__contains__('roc'):
            roc = data['roc']
        else:
            price = data['close']
            roc = ind.ROC(price, time_period=period)
        ax.plot(roc, label='roc%d' % period)
    drawHline(ax, hlines)
    ax.set_ylabel('ROC')


################################################################################################


def drawInd(ind='', ax=None, data=None):
    if ind == 'C':
        drawC(ax, data)
    if ind == 'K':
        drawK(ax, data)
    if ind == 'SMA':
        drawSMA(ax, data)
    if ind == 'BBANDS':
        drawBBANDS(ax, data)
    if ind == 'MINMAX':
        drawMINMAX(ax, data)
    if ind == 'WR':
        drawWR(ax, data)
    if ind == 'DMI':
        drawDMI(ax, data)
    if ind == 'KDJ':
        drawKDJ(ax, data)
    if ind == 'CCI':
        drawCCI(ax, data)
    if ind == 'MACD':
        drawMACD(ax, data)
    if ind == 'RSI':
        drawRSI(ax, data)
    if ind == 'EMV':
        drawEMV(ax, data)
    if ind == 'TRIX':
        drawTRIX(ax, data)
    if ind == 'MFI':
        drawMFI(ax, data)
    if ind == 'OBV':
        drawOBV(ax, data)
    if ind == 'ROC':
        drawROC(ax, data)


def drawAll(code, data, types=[['K', 'SMA']]):
    row = len(types)
    fig = plt.figure(figsize=(4, row))
    plt.title(code)
    i = 0
    for type in types:
        i += 1
        ax = fig.add_subplot(row, 1, i)

        j = 0
        for ind in type:
            j += 1
            if (i != 1) & (j == 2):
                ax = plt.twinx()
            drawInd(ind, ax, data)
            drawDate(ax, data)
            ax.legend(fontsize=9, ncol=3)

    plt.subplots_adjust(hspace=0.1)
    # plt.plot(CCI, 'k', lw=0.75, linestyle='-', label='CCI')
    # plt.legend(loc=2, prop={'size': 9.5})
    # plt.setp(plt.gca().get_xticklabels(), rotation=30)
    # plt.grid(True)
    plt.show()


def drawBuy(row, col, sector, symbols, start, end):
    fig = plt.figure(figsize=(16, 8))
    i = 0
    for symbol in symbols:
        i += 1
        data = TushareStore.get_usa_daily_data_ind(sector=sector, symbol=symbol, start_date=start,
                                                   end_date=end, append_ind=True)
        # data = store.get_chart_data_from_db("600%d.SH" % code, '20180101')
        # data = get_chart_data_from_web(code, '1/1/2018', '1/30/2019')
        close, pdi, wr, wr_89, bias = data['close'], data['pdi'], data['willr'], data['willr_89'], data['bias']
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

        # buy = close[(pdi <= 10) & (wr < -88)]
        # ax.scatter(buy.index, buy, c='red')
        # buy = close[(pdi <= 12) & (pdi > 10) & (wr < -88)]
        # ax.scatter(buy.index, buy, c='orange')
        # buy = close[(pdi <= 16) & (pdi > 12) & (wr < -88)]
        # ax.scatter(buy.index, buy, c='yellow')
        # buy = close[(pdi <= 20) & (pdi > 16) & (wr < -88)]
        # ax.scatter(buy.index, buy, c='green')

        # buy = close[ind.UP_CROSS(wr_89, -83.5) & ind.LESS_THAN(wr, -50)]
        # buy = close[ind.LESS_THAN(wr_89, -97)  & ind.BOTTOM(wr_89)]
        # buy = close[ind.LESS_THAN(bias, -12) & ind.BOTTOM(bias) & ind.LESS_THAN(wr_89, -83.5)]
        buy = close[ind.LESS_THAN(bias.shift(1), -13) & ind.BOTTOM(bias)]
        ax.scatter(buy.index, buy, s=20, c='green')
        # ax = plt.twinx()
        # ax.plot(bias)
        # ax.plot(wr)
        # drawHline(ax, [-12])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(symbol)

    plt.legend()
    plt.subplots_adjust(hspace=0.1)
    plt.show()


def drawPanel(row, col, sector, symbols, start, end):
    nasdaq = web.DataReader('^IXIC', start=start, end=end, data_source='yahoo')
    fig = plt.figure(figsize=(16, 8))
    i = 0
    for symbol in symbols:
        i += 1
        prices = store.get_usa_daily_data_ind(sector=sector, symbol=symbol, start_date=start, end_date=end)
        prices.index = prices.date
        # print prices
        ax = fig.add_subplot(row, col, i)
        ax.set_ylabel(symbol)
        ax.plot(prices['adj_close'])
        ax.set_xticks([])
        ax.set_yticks([])
        ax = ax.twinx()
        ax.plot(nasdaq['Adj Close'], color='grey')

    plt.show()


# code = 'GOOG'
# data = store.get_chart_data_from_db(code, '20180101')
# data = get_chart_data_from_web(code, '1/1/2018', '1/30/2019')
# K,SMA,WR,DMI,KDJ,CCI,RSI,MACD
# types = [['C', 'MINMAX'], ['C', 'WR'], ['C', 'DMI'], ['C', 'KDJ'], ['C', 'CCI'], ['C', 'RSI'], ['C', 'MACD']]
#
# types = [['C', 'MINMAX'], ['C', 'EMV'], ['C', 'TRIX'], ['C', 'OBV'], ['C', 'MFI'], ['C', 'RSI'], ['C', 'ROC']]
# drawAll(code, data, types=types)


################################################################################################
# a = np.arange(0, 12)
# # 501 516 598 589 582 586 589
# print np.reshape(a, (-1, 4)).shape[0]
# codes = ['AMZN', 'AAPL', 'GOOG', 'FB']
# codes = np.random.randint(100, 700, [12])
# print codes
#


def drawBuyA(row, col, symbols, start, end):
    fig = plt.figure(figsize=(16, 16))
    i = 0
    for symbol in symbols:
        print(symbol)
        i += 1
        data = TushareStore.get_a_daily_data_ind(table='a_daily', symbol=symbol, start_date=start,
                                                 end_date=end,
                                                 append_ind=True)
        close, pdi, wr, wr_89, bias = data['close'], data['pdi'], data['willr'], data['willr_89'], data['bias']
        ax = fig.add_subplot(row, col, i)
        ax.plot(close, c='grey')
        # buy = close[ind.LESS_THAN(bias.shift(1), -13) & ind.BOTTOM(bias)]
        buy = close[ind.LESS_THAN(wr.shift(1), -88) & ind.BOTTOM(wr)]
        # print buy
        ax.scatter(buy.index, buy, s=20, c='green')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(symbol)
    plt.legend()
    plt.subplots_adjust(hspace=1)
    plt.show()


if __name__ == '__main__':
    # df = pd.read_csv('CompanyList_technology_Cor.csv')
    # print df
    # df = df.iloc[0:120]
    symbols = TushareStore.get_a_stock_list('a_daily')
    symbols = symbols.iloc[:120, :]
    print(symbols)
    drawBuyA(12, 10, symbols['symbol'].tolist(), '2018-10-01', '2019-02-26')
