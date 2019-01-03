# -*- coding: utf-8 -*-
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import talib
import numpy as np


def MAX(price, time_period):
    return price.rolling(time_period).max()


def MIN(price, time_period):
    return price.rolling(time_period).min()


def STD(price, time_period):
    return price.rolling(time_period).std()


def HH(high, time_period):
    return MAX(high, time_period)


def LL(low, time_period):
    return MIN(low, time_period)


def TP(high, low, close):
    return (high + low + close) / 3


# Simple Moving Average (SMA) 简单移动平均线
# SMA: 10-period sum / 10
def SMA(price, time_period):
    return price.rolling(time_period).mean()


# Exponential Moving Average (EMA) 指数移动平均线
# Initial SMA: 10-period sum / 10
# Multiplier: (2 / (Time periods + 1) ) = (2 / (10 + 1) ) = 0.1818 (18.18%)
# EMA: {Close - EMA(previous day)} x multiplier + EMA(previous day).
def EMA(price, time_period):
    return price.ewm(ignore_na=False, span=time_period, min_periods=0, adjust=True).mean()


# Weighted Moving Average (WMA) 加权移动平均线
# Coppock Curve = 10-period WMA of (14-period RoC + 11-period RoC)
# WMA = Weighted Moving Average
# RoC = Rate-of-Change
def WMA(price, time_period):
    return price.rolling(time_period).apply(lambda x: np.average(x, weights=np.arange(1, time_period + 1)), raw=False)


def SMMA(price, time_period):
    return price.ewm(ignore_na=False, alpha=1.0 / time_period, min_periods=0, adjust=True).mean()


# Moving Average Convergence/Divergence Oscillator (MACD) 平滑异同移动平均线
# MACD: (12-day EMA - 26-day EMA)
# Signal Line: 9-day EMA of MACD
# MACD Histogram: MACD - Signal Line
def MACD(price, fast_period=12, slow_period=26, signal_period=9):
    macd = EMA(price, fast_period) - EMA(price, slow_period)
    macd_signal = EMA(macd, signal_period)
    macd_histogram = macd - macd_signal
    return macd, macd_signal, macd_histogram


# Bollinger Bands (BBANDS) 布林带
# Middle Band = 20-day simple moving average (SMA)
# Upper Band = 20-day SMA + (20-day standard deviation of price x 2)
# Lower Band = 20-day SMA - (20-day standard deviation of price x 2)
def BBANDS(price, time_period=5, nb_dev_up=2, nb_dev_dn=2):
    middle_band = SMA(price, time_period)
    upper_band = SMA(price, time_period) + STD(price, time_period) * nb_dev_up
    lower_band = SMA(price, time_period) - STD(price, time_period) * nb_dev_dn
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






# On Balance Volume (OBV) 能量潮指标
# If the closing price is above the prior close price then:
# Current OBV = Previous OBV + Current Volume
# If the closing price is below the prior close price then:
# Current OBV = Previous OBV  -  Current Volume
# If the closing prices equals the prior close price then:
# Current OBV = Previous OBV (no change)
def OBV(price, volume):
    pnv = volume.where(price > price.shift(1), -volume)[3:]
    return pnv.cumsum()


# Typical Price = (High + Low + Close)/3
# Raw Money Flow = Typical Price x Volume
# Money Flow Ratio = (14-period Positive Money Flow)/(14-period Negative Money Flow)
# Money Flow Index = 100 - 100/(1 + Money Flow Ratio)
def MFI(high, low, close, volume, time_period):
    tp = TP(high, low, close)
    raw_money_flow = tp * volume
    pos_money_flow = raw_money_flow.where(tp > tp.shift(1), 0)
    neg_money_flow = raw_money_flow.where(tp < tp.shift(1), 0)
    pos_money_flow_sum = pos_money_flow.rolling(time_period).sum()
    neg_money_flow_sum = neg_money_flow.rolling(time_period).sum()
    money_flow_ratio = pos_money_flow_sum / neg_money_flow_sum
    money_flow_index = 100 - 100 / (1 + money_flow_ratio)
    return money_flow_index


# Aroon-Up = ((25 - Days Since 25-day High)/25) x 100
# Aroon-Down = ((25 - Days Since 25-day Low)/25) x 100
def AROON(high, low, time_period):
    aroon_up = high.rolling(time_period + 1).apply(
        lambda h: pd.Series(h).idxmax() * 100.0 / time_period, raw='False')
    aroon_down = low.rolling(time_period + 1).apply(
        lambda l: pd.Series(l).idxmin() * 100.0 / time_period, raw='False')
    return aroon_up, aroon_down


# Aroon Oscillator = Aroon-Up  -  Aroon-Down
def AROONOSC(high, low, time_period):
    aroon_up, aroon_down = AROON(high, low, time_period)
    return aroon_up - aroon_down


# Percentage Price Oscillator (PPO) 价格震荡百分比指数 和talib有出入
# Percentage Price Oscillator (PPO): {(12-day EMA - 26-day EMA)/26-day EMA} x 100
# Signal Line: 9-day EMA of PPO
# PPO Histogram: PPO - Signal Line
def PPO(price, fast_period=12, slow_period=26, signal_period=9):
    ppo = (EMA(price, fast_period) - EMA(price, slow_period)) * 100.0 / EMA(price, slow_period)
    ppo_signal = EMA(ppo, signal_period)
    ppo_histogram = ppo - ppo_signal
    return ppo, ppo_signal, ppo_histogram





# StochRSI = (RSI - Lowest Low RSI) / (Highest High RSI - Lowest Low RSI)
def STOCHRSI(data, time_period=14):
    rsi = RSI(data, time_period)
    ll_rsi = LL(rsi, time_period)
    hh_rsi = HH(rsi, time_period)
    return (rsi - ll_rsi) / (hh_rsi - ll_rsi)


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
    hh, ll = HH(h, fastk_period), LL(l, fastk_period)
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
    hh, ll = HH(h, fastk_period), LL(l, fastk_period)
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
def TR1(high, low, close):
    df = pd.DataFrame()
    df['hl'] = (high - low).abs()
    df['hcl'] = (close.shift(1) - high).abs()
    df['cll'] = (close.shift(1) - low).abs()
    return df.max(axis=1)


def TR(data):
    return TR(data['High'], data['Low'], data['Close'])


# ATR : SMMA(TR,N)
def ATR(data, time_period):
    return SMMA(TR(data), time_period)


# HD = MAX(H-REF(H),0)
# LD = MAX(REF(L)-L,0)
# +DM = SMMA(HD>LD?HD:0)
# -DM = SMMA(HD<LD?LD:0)
def _DM_(data, time_period):
    h, l = data['High'], data['Low']
    df = pd.DataFrame()
    hd = h - h.shift(1)
    hd = (hd + hd.abs()) / 2
    ld = l.shift(1) - l
    ld = (ld + ld.abs()) / 2
    tr = df['tr'] = TR(data)  # 去掉这行报错，不知为什么

    df['pdm'] = np.where(hd > ld, hd, 0)
    df['mdm'] = np.where(hd < ld, ld, 0)
    pdm_smma = SMMA(df['pdm'], time_period)  # +DM
    mdm_smma = SMMA(df['mdm'], time_period)  # -DM
    return pdm_smma, mdm_smma


def DM(data, time_period):
    pdm_smma, mdm_smma = _DM_(data, time_period)
    return pdm_smma * time_period, mdm_smma * time_period


# +DI = +DM/TR*100
# -DI = -DM/TR*100
def DI(data, time_period):
    pdm, mdm = _DM_(data, time_period)
    tr = SMMA(TR(data), time_period)
    pdi = pdm / tr * 100  # +DI
    mdi = mdm / tr * 100  # -DI
    return pdi, mdi


# DX = (DI DIF/DI SUM)*100
# DX = |(+DI14)-(-DI14)|/|(+DI14)+(-DI14)|
def DX(data, time_period):
    pdi, mdi = DI(data, time_period)
    return (pdi - mdi).abs() / (pdi + mdi) * 100  # DX


# ADX = SMMA(DX,N)
def ADX(data, time_period):
    return SMMA(DX(data, time_period), time_period)


# ADXR = (ADX+REF(ADX,N))/2
def ADXR(data, time_period):
    return (ADX(data, time_period) + ADX(data, time_period).shift(time_period)) / 2

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


# %R = (Highest High - Close)/(Highest High - Lowest Low) * -100
# Lowest Low = lowest low for the look-back period
# Highest High = highest high for the look-back period
# %R is multiplied by -100 correct the inversion and move the decimal.
def WILLR(data, time_period):
    h, l, c = data['High'], data['Low'], data['Close']
    hh, ll = HH(h, time_period), LL(l, time_period)
    return (hh - c) / (hh - ll) * (-100)


# def TR(data, time_period):
#     return EMA(EMA(EMA(data, time_period), time_period), time_period)


def TRIX(data, time_period):
    tr = EMA(EMA(EMA(data, time_period), time_period), time_period)
    return (tr - tr.shift(1)) / tr.shift(1) * 100


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

# ppo, ppo_signal, ppo_hist = PPO(data['Close'], fast_period=12, slow_period=26, signal_period=9)
# ax.plot(ppo, label="talib_macd")
# ax.plot(ppo_signal, label="talib_macd_signal")
# ax.plot(data.index, ppo)
# ax.plot(data.index, ppo_signal)
# ax.plot(data.index, ppo_hist)

# ppo_hist1 = talib.PPO(data['Close'], fastperiod=12, slowperiod=26, matype=1)
# ax.plot(data.index, ppo_hist1)

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

# ax.plot(ADXR(data, time_period=14))
# ax.plot(talib.ADXR(data['High'], data['Low'], data['Close'], timeperiod=14))

# pdm, mdm = DM(data, time_period=14)
# ax.plot(pdm)
# ax.plot(mdm)
# ax.plot(talib.MINUS_DM(data['High'], data['Low'], timeperiod=14))
# ax.plot(talib.PLUS_DM(data['High'], data['Low'], timeperiod=14))
#
# ax.plot(DX(data, time_period=14))
# ax.plot(talib.DX(data['High'], data['Low'], data['Close'], timeperiod=14))

# pdi, mdi = DI(data, time_period=14)
# ax.plot(pdi)
# ax.plot(mdi)
# ax.plot(talib.PLUS_DI(data['High'], data['Low'], data['Close'], timeperiod=14))
# ax.plot(talib.MINUS_DI(data['High'], data['Low'], data['Close'], timeperiod=14))

# ax.plot(WILLR(data, time_period=14))
# ax.plot(talib.WILLR(data['High'], data['Low'], data['Close'], timeperiod=14))
# ax.plot(TRIX(data['Close'], time_period=14))
# ax.plot(talib.TRIX(data['Close'], timeperiod=14))
# ax.plot(STOCHRSI(data['Close'], time_period=14))
# ax.plot(talib.STOCHRSI(data['Close'], timeperiod=14))

# ax.plot(AROON(data['High'], data['Low'], time_period=14))
# h, l = AROON(data['High'], data['Low'], time_period=14)
# h1, l1 = talib.AROON(data['High'], data['Low'], timeperiod=14)
# ax.plot(h)
# ax.plot(l)
# ax.plot(h1)
# ax.plot(l1)

# ax.plot(AROONOSC(data['High'], data['Low'], time_period=14))
# ax.plot(talib.AROONOSC(data['High'], data['Low'], timeperiod=14))

# ax.plot(MFI(data['High'], data['Low'], data['Close'], data['Volume'], time_period=14))
# ax.plot(talib.MFI(data['High'], data['Low'], data['Close'], data['Volume'], timeperiod=14))

# ax.plot(OBV(data['High'], data['Volume']))
# ax.plot(talib.OBV(data['High'], data['Volume']))

plt.legend()
plt.show()
# print ADX(data, time_period=14)
