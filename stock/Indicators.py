# -*- coding: utf-8 -*-
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import talib
import numpy as np

#https://github.com/jealous/stockstats/blob/master/stockstats.py
# Simple Moving Average (SMA) 简单移动平均线
# SMA: 10-period sum / 10
def SMA(data, time_period):
    return data.rolling(time_period).mean()


# Exponential Moving Average (EMA) 指数移动平均线
# Initial SMA: 10-period sum / 10
# Multiplier: (2 / (Time periods + 1) ) = (2 / (10 + 1) ) = 0.1818 (18.18%)
# EMA: {Close - EMA(previous day)} x multiplier + EMA(previous day).
def EMA(data, time_period):
    return data.ewm(ignore_na=False, span=time_period, min_periods=0, adjust=True).mean()


# Weighted Moving Average (WMA)
# Coppock Curve = 10-period WMA of (14-period RoC + 11-period RoC)
# WMA = Weighted Moving Average
# RoC = Rate-of-Change
def WMA(data, time_period):
    return data.rolling(time_period).apply(lambda x: np.average(x, weights=np.arange(1, time_period + 1)))


def SMMA(data, time_period):
    return data.ewm(ignore_na=False, alpha=1.0 / time_period, min_periods=0, adjust=True).mean()


# Moving Average Convergence/Divergence Oscillator (MACD) 平滑异同移动平均线
# MACD: (12-day EMA - 26-day EMA)
# Signal Line: 9-day EMA of MACD
# MACD Histogram: MACD - Signal Line
def MACD(data, fast_period=12, slow_period=26, signal_period=9):
    macd = EMA(data, fast_period) - EMA(data, slow_period)
    macd_signal = EMA(macd, signal_period)
    macd_histogram = macd - macd_signal
    return macd, macd_signal, macd_histogram


# Bollinger Bands (BBANDS) 布林带
# Middle Band = 20-day simple moving average (SMA)
# Upper Band = 20-day SMA + (20-day standard deviation of price x 2)
# Lower Band = 20-day SMA - (20-day standard deviation of price x 2)
def BBANDS(data, time_period=5, nbdevup=2, nbdevdn=2):
    middle_band = SMA(data, time_period)
    upper_band = SMA(data, time_period) + data.rolling(time_period).std() * nbdevup
    lower_band = SMA(data, time_period) - data.rolling(time_period).std() * nbdevdn
    return upper_band, middle_band, lower_band


# Relative Strength Index (RSI) 相对强弱指数（0~100）
#               100
# RSI = 100 - --------
#              1 + RS
# RS = Average Gain / Average Loss
# First Average Gain = Sum of Gains over the past 14 periods / 14.
# First Average Loss = Sum of Losses over the past 14 periods / 14
# Average Gain = [(previous Average Gain) x 13 + current Gain] / 14.
# Average Loss = [(previous Average Loss) x 13 + current Loss] / 14.
def RSI(data, time_period=14):
    diff = data.diff(1)
    alpha = 1.0 / time_period
    avg_gain = diff.clip_lower(0).ewm(alpha=alpha).mean()
    avg_loss = diff.clip_upper(0).ewm(alpha=alpha).mean()
    rsi = 100 - 100 / (1 - avg_gain / avg_loss)
    return rsi


# KDJ
# %K = (Current Close - Lowest Low)/(Highest High - Lowest Low) * 100
# %D = 3-day SMA of %K
# Lowest Low = lowest low for the look-back period
# Highest High = highest high for the look-back period
# %K is multiplied by 100 to move the decimal point two places
# Fast Stochastic Oscillator:
# Fast %K = %K basic calculation
# Fast %D = 3-period SMA of Fast %K

def STOCHF(data, fastk_period=5, fastd_period=3):
    h, l, c = data['High'], data['Low'], data['Close']
    hh, ll = h.rolling(fastk_period).max(), l.rolling(fastk_period).min()
    fast_k = (c - ll) / (hh - ll) * 100
    fast_d = SMA(fast_k, fastd_period)
    return fast_k, fast_d


# Slow Stochastic Oscillator:
# Slow %K = Fast %K smoothed with 3-period SMA
# Slow %D = 3-period SMA of Slow %K
# Full Stochastic Oscillator:
# Full %K = Fast %K smoothed with X-period SMA
# Full %D = X-period SMA of Full %K
# J = 3 * K - 2 * D(v)
# J = 3 * D - 2 * K
def STOCH(data, fastk_period=5, slowk_period=3, slowd_period=3):
    h, l, c = data['High'], data['Low'], data['Close']
    hh, ll = h.rolling(fastk_period).max(), l.rolling(fastk_period).min()
    fast_k = (c - ll) / (hh - ll) * 100
    slow_k = SMA(fast_k, slowk_period)
    slow_d = SMA(slow_k, slowd_period)
    return slow_k, slow_d


# Commodity Channel Index (CCI) 顺势指标   算法与talib有出入
# CCI = (Typical Price  -  Time period SMA of TP) / (.015 x  Time period Mean Deviation of TP)
# Typical Price (TP) = (High + Low + Close)/3
# Constant = .015
def CCI(data, time_period=20):
    h, l, c = data['High'], data['Low'], data['Close']
    # hh, ll = h.rolling(time_period).max(), l.rolling(time_period).min()
    tp = (h + l + c) / 3
    tp_sma = SMA(tp, time_period)
    tp_std = tp.rolling(time_period).std()
    return (tp - tp_sma) / (.015 * tp_std)


# ROC - Rate of change 变动率指标
# ROC = [(Close - Close n periods ago) / (Close n periods ago)] * 100
def ROC(data, time_period):
    c = data['Close']
    return (c - c.shift(time_period)) / c.shift(time_period) * 100


# Ease of Movement (EMV) 简易波动指标 talib没有
# Distance Moved = ((H + L)/2 - (Prior H + Prior L)/2)
# Box Ratio = ((V/100,000,000)/(H - L))
# 1-Period EMV = ((H + L)/2 - (Prior H + Prior L)/2) / ((V/100,000,000)/(H - L))
# 14-Period Ease of Movement = 14-Period simple moving average of 1-period EMV
def EMV(data, time_period=14):
    h, l, v = data['High'], data['Low'], data['Volume']
    distance_moved = (h + l) / 2 - (h.shift(1) + l.shift(1)) / 2
    box_ratio = ((v / 100000000) / (h - l))
    emv = distance_moved / box_ratio
    return emv.rolling(time_period).mean()


# TR : MAX(MAX((HIGH-LOW),ABS(REF(CLOSE,1)-HIGH)),ABS(REF(CLOSE,1)-LOW))
def TR(data):
    h, l, c = data['High'], data['Low'], data['Close']
    df = pd.DataFrame()
    df['hl'] = (h - l).abs()
    df['hcl'] = (c.shift(1) - h).abs()
    df['cll'] = (c.shift(1) - l).abs()
    return df.max(axis=1)


# ATR : SMMA(TR,N)
def ATR(data, time_period):
    return SMMA(TR(data), time_period)


def ADX(data, time_period):
    h, l, v = data['High'], data['Low'], data['Volume']
    df = pd.DataFrame()
    hd = h - h.shift(1)
    hd = (hd + hd.abs()) / 2
    ld = l.shift(1) - l
    ld = (ld + ld.abs()) / 2
    tr = df['tr'] = TR(data)

    df['pdm'] = np.where(hd > ld, hd, 0)
    df['mdm'] = np.where(hd < ld, ld, 0)
    pdm_smma = SMMA(df['pdm'], time_period)
    mdm_smma = SMMA(df['mdm'], time_period)
    tr_smma = SMMA(tr, time_period)
    pdi = pdm_smma / tr_smma * 100
    mdi = mdm_smma / tr_smma * 100
    dx = (pdi - mdi).abs() / (pdi + mdi) * 100
    return SMMA(dx, time_period)

    # df['hd'] = h - h.shift(1)
    # df.ix[df['hd'] < 0, 'hd'] = 0
    # df['ld'] = l.shift(1) - l
    # df.ix[df['ld'] < 0, 'ld'] = 0
    # df['tr'] = TR(data)
    #
    # df.ix[df['hd'] > df['ld'], 'pdm'] = df['hd']
    # df.ix[df['hd'] < df['ld'], 'mdm'] = df['ld']
    # df['pdm'].fillna(0, inplace=True)
    # df['mdm'].fillna(0, inplace=True)
    #
    # df['pdm%d' % time_period] = SMMA(df['pdm'], time_period=time_period)
    # df['mdm%d' % time_period] = SMMA(df['mdm'], time_period=time_period)
    # df['tr%d' % time_period] = SMMA(df['tr'], time_period=time_period)
    #
    # df['pdi%d' % time_period] = df['pdm%d' % time_period] / df['tr%d' % time_period] * 100
    # df['mdi%d' % time_period] = df['mdm%d' % time_period] / df['tr%d' % time_period] * 100
    # df['dx'] = ((df['pdi%d' % time_period] - df['mdi%d' % time_period]) / (
    #         df['pdi%d' % time_period] + df['mdi%d' % time_period])).abs() * 100
    # return SMMA(df['dx'], time_period)


data = web.DataReader('GOOG', data_source='yahoo', start='9/1/2018', end='12/30/2018')
data = pd.DataFrame(data)
Close = data['Close']

# m, s, h = MACD(Close, fast_period=12, slow_period=26, signal_period=9)
# macd, macdsignal, macdhist = talib.MACD(Close, fastperiod=12, slowperiod=26, signalperiod=9)

# u, m, l = BBANDS(Close, 20)
# upper, middle, lower = talib.BBANDS(Close, 20)

wma = WMA(Close, time_period=5)
wma1 = talib.WMA(Close, timeperiod=5)

rsi = RSI(Close, time_period=14)
rsi2 = talib.RSI(Close, 14)

f, k = STOCH(data, fastk_period=5, slowk_period=3, slowd_period=3)
f1, k1 = talib.STOCH(data['High'], data['Low'], data['Close'], fastk_period=5, slowk_period=3, slowd_period=3)

f, k = STOCHF(data, fastk_period=5, fastd_period=3)
f1, k1 = talib.STOCHF(data['High'], data['Low'], data['Close'], fastk_period=5, fastd_period=3)

roc = ROC(data, time_period=10)
roc1 = talib.ROC(data['Close'], timeperiod=10)

cci = CCI(data, time_period=14)
cci1 = talib.CCI(data['High'], data['Low'], data['Close'], timeperiod=14)

emv = EMV(data, time_period=14)

# atr = ATR(data, time_period=14)
atr1 = talib.ATR(data['High'], data['Low'], data['Close'], timeperiod=14)

fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(2, 1, 1)

# ax.plot(ATR(data, time_period=14))
# ax.plot(WMA(TR(data), time_period=14))
# ax.plot(SMA(TR(data), time_period=14))
# ax.plot(EMA(TR(data), time_period=14))
# ax.plot(atr1, label='t')
# ax.plot(wma)
# ax.plot(wma1)
# ax.plot(m)
# ax.plot(s)
# ax.bar(data.index, h)
# ax.plot(macd, label="talib_macd")
# ax.plot(macdsignal, label="talib_macd_signal")
# ax.bar(data.index, macdhist)

# ax.plot(u)
# ax.plot(m)
# ax.plot(l)
#
# ax.plot(upper)
# ax.plot(middle)
# ax.plot(lower)

# ax.plot(rsi, label="rsi")
# ax.plot(rsi2)
# ax.plot(f)
# ax.plot(k)
# ax.plot(f1)
# ax.plot(k1)

# ax.plot(roc)
# ax.plot(roc1)

# ax.plot(cci, label="cci")
# ax.plot(cci1)

# ax.plot(emv, label="emv")

ax.plot(ADX(data, time_period=14))
ax.plot(talib.ADX(data['High'], data['Low'], data['Close'], timeperiod=14))

plt.legend()
plt.show()
print ADX(data, time_period=14)
