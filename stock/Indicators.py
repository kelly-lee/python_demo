# -*- coding: utf-8 -*-
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import talib
import numpy as np


def DIFF(price, time_period=1):
    return price.diff(time_period)


def REF(price, time_period=1):
    return price.shift(time_period)


def MAX(price, time_period):
    return price.rolling(time_period).max()


def MAXINDEX(price, time_period):
    return price.rolling(time_period).apply(lambda h: pd.Series(h).idxmax(), raw=True)


def MIN(price, time_period):
    return price.rolling(time_period).min()


def MININDEX(price, time_period):
    return price.rolling(time_period).apply(lambda h: pd.Series(h).idxmin(), raw=True)


def STD(price, time_period):
    return price.rolling(time_period).std()


def SUM(price, time_period):
    return price.rolling(time_period).sum()


def MEDIAN(high, low):
    return (high + low) / 2


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


# Stochastic Oscillator (KD) 随机指标
# %K = (Current Close - Lowest Low)/(Highest High - Lowest Low) * 100
# %D = 3-day SMA of %K
# Lowest Low = lowest low for the look-back period
# Highest High = highest high for the look-back period
# %K is multiplied by 100 to move the decimal point two places
# Fast Stochastic Oscillator:
# Fast %K = %K basic calculation
# Fast %D = 3-period SMA of Fast %K
def STOCHF(high, low, close, fastk_period=5, fastd_period=3):
    hh, ll = HH(high, fastk_period), LL(low, fastk_period)
    fast_k = (close - ll) / (hh - ll) * 100
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
def STOCH(high, low, close, fastk_period=5, slowk_period=3, slowd_period=3):
    hh, ll = HH(high, fastk_period), LL(low, fastk_period)
    fast_k = (close - ll) / (hh - ll) * 100
    slow_k = SMA(fast_k, slowk_period)
    slow_d = SMA(slow_k, slowd_period)
    return slow_k, slow_d


# Relative Strength Index (RSI) 相对强弱指数（0~100）
#               100
# RSI = 100 - --------
#              1 + RS
# RS = Average Gain / Average Loss
# First Average Gain = Sum of Gains over the past 14 periods / 14.
# First Average Loss = Sum of Losses over the past 14 periods / 14
# Average Gain = [(previous Average Gain) x 13 + current Gain] / 14.
# Average Loss = [(previous Average Loss) x 13 + current Loss] / 14.
def RSI(price, time_period=14):
    diff = DIFF(price)
    avg_gain = SMMA(diff.clip_lower(0), time_period)
    avg_loss = SMMA(diff.clip_upper(0), time_period)
    rsi = 100 - 100 / (1 - avg_gain / avg_loss)
    return rsi


# Commodity Channel Index (CCI) 顺势指标   算法与talib有出入
# CCI = (Typical Price  -  Time period SMA of TP) / (.015 x  Time period Mean Deviation of TP)
# Typical Price (TP) = (High + Low + Close)/3
# Constant = .015
def CCI(high, low, close, time_period=20):
    tp = TP(high, low, close)
    tp_sma = SMA(tp, time_period)
    tp_std = STD(tp, time_period)
    return (tp - tp_sma) / (.015 * tp_std)


# ROC - Rate of change 变动率指标
# ROC = [(Close - Close n periods ago) / (Close n periods ago)] * 100
def ROC(price, time_period):
    return DIFF(price, time_period) / REF(price, time_period) * 100


# Typical Price = (High + Low + Close)/3
# Raw Money Flow = Typical Price x Volume
# Money Flow Ratio = (14-period Positive Money Flow)/(14-period Negative Money Flow)
# Money Flow Index = 100 - 100/(1 + Money Flow Ratio)
def MFI(high, low, close, volume, time_period):
    tp = TP(high, low, close)
    raw_money_flow = tp * volume
    pos_money_flow = raw_money_flow.where(tp > tp.shift(1), 0)
    neg_money_flow = raw_money_flow.where(tp < tp.shift(1), 0)
    money_flow_ratio = SUM(pos_money_flow, time_period) / SUM(neg_money_flow, time_period)
    return 100 - 100 / (1 + money_flow_ratio)


# %R = (Highest High - Close)/(Highest High - Lowest Low) * -100
# Lowest Low = lowest low for the look-back period
# Highest High = highest high for the look-back period
# %R is multiplied by -100 correct the inversion and move the decimal.
def WILLR(high, low, close, time_period):
    hh, ll = HH(high, time_period), LL(low, time_period)
    return (hh - close) / (hh - ll) * (-100)


# def TR(data, time_period):
#     return EMA(EMA(EMA(data, time_period), time_period), time_period)
def TRIX(price, time_period):
    tr = EMA(EMA(EMA(price, time_period), time_period), time_period)
    return (tr - REF(tr)) / REF(tr) * 100


# Aroon-Up = ((25 - Days Since 25-day High)/25) x 100
# Aroon-Down = ((25 - Days Since 25-day Low)/25) x 100
def AROON(high, low, time_period):
    aroon_up = MAXINDEX(high, time_period + 1) * 100 / time_period
    aroon_down = MININDEX(low, time_period + 1) * 100 / time_period
    return aroon_up, aroon_down


# Aroon Oscillator = Aroon-Up  -  Aroon-Down
def AROONOSC(high, low, time_period):
    aroon_up, aroon_down = AROON(high, low, time_period)
    return aroon_up - aroon_down

# Ease of Movement (EMV) 简易波动指标 talib没有
# Distance Moved = ((H + L)/2 - (Prior H + Prior L)/2)
# Box Ratio = ((V/100,000,000)/(H - L))
# 1-Period EMV = ((H + L)/2 - (Prior H + Prior L)/2) / ((V/100,000,000)/(H - L))
# 14-Period Ease of Movement = 14-Period simple moving average of 1-period EMV
def EMV(high, low, volume, time_period=14):
    distance_moved = MEDIAN(high, low) - MEDIAN(REF(high), REF(low))
    box_ratio = (volume / 100000000) / (high - low)
    emv = distance_moved / box_ratio
    return SMA(emv, time_period)


###########################################################################


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


# TR : MAX(MAX((HIGH-LOW),ABS(REF(CLOSE,1)-HIGH)),ABS(REF(CLOSE,1)-LOW))
def TR1(high, low, close):
    df = pd.DataFrame()
    df['hl'] = (high - low).abs()
    df['hcl'] = (close.shift(1) - high).abs()
    df['cll'] = (close.shift(1) - low).abs()
    return df.max(axis=1)


def TR(data):
    return TR1(data['High'], data['Low'], data['Close'])


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
def DX(price, time_period):
    pdi, mdi = DI(price, time_period)
    return (pdi - mdi).abs() / (pdi + mdi) * 100.0  # DX


# ADX = SMMA(DX,N)
def ADX(price, time_period):
    return SMMA(DX(price, time_period), time_period)


# ADXR = (ADX+REF(ADX,N))/2
def ADXR(price, time_period):
    adx = ADX(price, time_period)
    return (adx + REF(adx, time_period)) / 2

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


data = web.DataReader('GOOG', data_source='yahoo', start='6/1/2018', end='12/30/2018')
data = pd.DataFrame(data)
high, low, close, volume = data['High'], data['Low'], data['Close'], data['Volume']
print 'load data'
fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(2, 1, 1)

# ax.plot(SMA(close, time_period=5))
# ax.plot(talib.SMA(close, timeperiod=5))

# ax.plot(WMA(close, time_period=5))
# ax.plot(talib.WMA(close, timeperiod=5))

# ax.plot(EMA(close, time_period=12))
# ax.plot(talib.EMA(close, timeperiod=12))
# ax.plot(EMA(close, time_period=26))
# ax.plot(talib.EMA(close, timeperiod=26))

# ax.plot(EMA(close, time_period=12) - EMA(close, time_period=26))
# ax.plot(talib.EMA(close, timeperiod=12) - talib.EMA(close, timeperiod=26))

# u, m, l = BBANDS(close, 20)
# upper, middle, lower = talib.BBANDS(close, 20)
# ax.plot(u)
# ax.plot(m)
# ax.plot(l)
# ax.plot(upper)
# ax.plot(middle)
# ax.plot(lower)

# m, s, h = MACD(close, fast_period=12, slow_period=26, signal_period=9)
# macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
# ax.plot(m)
# ax.plot(s)
# ax.bar(data.index, h)
# ax.plot(macd)
# ax.plot(macdsignal)
# ax.bar(data.index, macdhist)

# PPO有问题********************
# p, p_s, p_h = PPO(close, fast_period=12, slow_period=26, signal_period=9)
# ppo_hist = talib.PPO(close, fastperiod=12, slowperiod=26)
# ax.plot(p)
# ax.plot(p_s)
# ax.plot(p_h)
# ax.plot(data.index, ppo_hist)

# s_k, s_d = STOCH(high, low, close, fastk_period=5, slowk_period=3, slowd_period=3)
# slow_k, slow_d = talib.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowd_period=3)
# ax.plot(s_k)
# ax.plot(s_d)
# ax.plot(slow_k)
# ax.plot(slow_d)

# f_k, f_d = STOCHF(high, low, close, fastk_period=5, fastd_period=3)
# fast_k, fast_d = talib.STOCHF(high, low, close, fastk_period=5, fastd_period=3)
# ax.plot(f_k)
# ax.plot(f_d)
# ax.plot(fast_k)
# ax.plot(fast_d)

# ax.plot(RSI(close, time_period=14))
# ax.plot(talib.RSI(close, timeperiod=14))

# CCI有差别********************
# ax.plot(CCI(high, low, close, time_period=14))
# ax.plot(talib.CCI(high, low, close, timeperiod=14))

# ax.plot(ROC(close, time_period=10))
# ax.plot(talib.ROC(close, timeperiod=10))

# ax.plot(MFI(high, low, close, volume, time_period=14))
# ax.plot(talib.MFI(high, low, close, volume, timeperiod=14))

# ax.plot(WILLR(high, low, close, time_period=14))
# ax.plot(talib.WILLR(high, low, close, timeperiod=14))

# ax.plot(TRIX(close, time_period=14))
# ax.plot(talib.TRIX(close, timeperiod=14))

# a_u, a_d = AROON(high, low, time_period=14)
# aroon_up, aroon_down = talib.AROON(high, low, timeperiod=14)
# ax.plot(a_u)
# ax.plot(a_d)
# ax.plot(aroon_up)
# ax.plot(aroon_down)

# ax.plot(EMV(high, low, volume, time_period=14), label="emv")
##################################################################################################


# atr = ATR(data, time_period=14)
# atr1 = talib.ATR(data['High'], data['Low'], data['Close'], timeperiod=14)

# ax.plot(ATR(data, time_period=14))

# ax.plot(atr1, label='t')


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


# ax.plot(STOCHRSI(data['Close'], time_period=14))
# ax.plot(talib.STOCHRSI(data['Close'], timeperiod=14))

# ax.plot(AROON(data['High'], data['Low'], time_period=14))


# ax.plot(AROONOSC(data['High'], data['Low'], time_period=14))
# ax.plot(talib.AROONOSC(data['High'], data['Low'], timeperiod=14))


# ax.plot(OBV(data['High'], data['Volume']))
# ax.plot(talib.OBV(data['High'], data['Volume']))

plt.legend()
plt.show()
# print ADX(data, time_period=14)
