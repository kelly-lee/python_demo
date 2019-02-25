# -*- coding: utf-8 -*-
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import talib
import numpy as np


# class Indicator:
#     def __init__(self, data):
#         self.data = data
#         self.close = data['Close']
#
#     def DIFF(self, price=None, time_period=1):
#         if price is None:
#             return self.close.diff(time_period)
#         else:
#             return self.data[price].diff(time_period)
#
#
# data = web.DataReader('GOOG', data_source='yahoo', start='1/1/2017', end='1/30/2019')
# data = pd.DataFrame(data)
#
# ind = Indicator(data)
# print DIFF()

def LESS_THAN(price, line):
    return price <= line


def GREAT_THAN(price, line):
    return price >= line


def BETWEEN(price, low, high):
    return (price >= low) & (price <= high)


def UP_CROSS(price, line):
    return (price.shift(1) <= line) & (price >= line)


def DOWN_CROSS(price, line):
    return (price.shift(1) >= line) & (price <= line)


def GOLDEN_CROSS(fast, slow):
    return (fast.shift(1) <= slow.shift(1)) & (fast >= slow)


def DEAD_CROSS(fast, slow):
    return (fast.shift(1) >= slow.shift(1)) & (fast <= slow)


def TOP(price):
    return (price.shift(2) <= price.shift(1)) & (price.shift(1) >= price)


def BOTTOM(price):
    return (price.shift(2) >= price.shift(1)) & (price.shift(1) <= price)


###############################################################################

def DIFF(price, time_period=1):
    return price.diff(time_period)


def REF(price, time_period=1):
    return price.shift(time_period)


def ABS(price):
    return price.abs()


def PCT_CHANGE(price, time_period=1):
    return price.pct_change(time_period)


def DRCT(price, time_period=1):
    return PCT_CHANGE(price, time_period).apply(lambda p: np.sign(p))


def MEDIAN(high, low):
    return (high + low) / 2


def TP(high, low, close):
    return (high + low + close) / 3


######################################################################################

def MAX(price, time_period):
    return price.rolling(time_period).max()


def MAXINDEX(price, time_period):
    return price.rolling(time_period).apply(lambda p: pd.Series(p).idxmax(), raw=True)


def MIN(price, time_period):
    return price.rolling(time_period).min()


def MININDEX(price, time_period):
    return price.rolling(time_period).apply(lambda p: pd.Series(p).idxmin(), raw=True)


def HH(high, time_period):
    return MAX(high, time_period)


def LL(low, time_period):
    return MIN(low, time_period)


def STD(price, time_period):
    return price.rolling(time_period).std()


def SUM(price, time_period):
    return price.rolling(time_period).sum()


######################################################################################

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


# 双指数移动平均线技术指标
# MA_Type: 0=SMA, 1=EMA, 2=WMA, 3=DEMA, 4=TEMA, 5=TRIMA, 6=KAMA, 7=MAMA, 8=T3 (Default=SMA)
def DEMA(price, time_period):
    return 2 * EMA(price, time_period) - EMA(EMA(price, time_period), time_period)


# 3 * EMA(Price, N, i) - 3 * EMA2(Price, N, i) + EMA3(Price, N, i)
def TEMA(price, time_period):
    return 3 * EMA(price, time_period) - 3 * EMA(EMA(price, time_period), time_period) + EMA(
        EMA(EMA(price, time_period), time_period), time_period)


# Bollinger Bands (BBANDS) 布林带
# Middle Band = 20-day simple moving average (SMA)
# Upper Band = 20-day SMA + (20-day standard deviation of price x 2)
# Lower Band = 20-day SMA - (20-day standard deviation of price x 2)
def BBANDS(price, time_period=5, nb_dev_up=2, nb_dev_dn=2):
    middle_band = SMA(price, time_period)
    upper_band = SMA(price, time_period) + STD(price, time_period) * nb_dev_up
    lower_band = SMA(price, time_period) - STD(price, time_period) * nb_dev_dn
    return upper_band, middle_band, lower_band


######################################################################################

# Moving Average Convergence/Divergence Oscillator (MACD) 平滑异同移动平均线
# MACD: (12-day EMA - 26-day EMA)
# Signal Line: 9-day EMA of MACD
# MACD Histogram: MACD - Signal Line
def MACD(price, fast_period=12, slow_period=26, signal_period=9):
    macd = EMA(price, fast_period) - EMA(price, slow_period)
    macd_signal = EMA(macd, signal_period)
    macd_histogram = macd - macd_signal
    return macd, macd_signal, macd_histogram


# %R = (Highest High - Close)/(Highest High - Lowest Low) * -100
# Lowest Low = lowest low for the look-back period
# Highest High = highest high for the look-back period
# %R is multiplied by -100 correct the inversion and move the decimal.
def WILLR(high, low, close, time_period):
    hh, ll = HH(high, time_period), LL(low, time_period)
    return (hh - close) / (hh - ll) * (-100)


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


# TR : MAX(MAX((HIGH-LOW),ABS(REF(CLOSE,1)-HIGH)),ABS(REF(CLOSE,1)-LOW))
def TR(high, low, close):
    df = pd.DataFrame()
    df['hl'] = ABS(high - low)
    df['hcl'] = ABS(REF(close) - high)
    df['cll'] = ABS(REF(close) - low)
    return df.max(axis=1)


# ATR : SMMA(TR,N)
def ATR(high, low, close, time_period):
    return SMMA(TR(high, low, close), time_period)


# HD = HIGH-REF(HIGH,1)
# LD = REF(LOW,1)-LOW
# +DM = SMMA(IF(HD>0 && HD>LD,HD,0),N)
# -DM = SMMA(IF(LD>0 && LD>HD,LD,0),N)
def _DM_(high, low, time_period):
    hd = DIFF(high)
    ld = -DIFF(low)
    pdm = SMMA(hd.where((hd > 0) & (hd > ld), 0), time_period)  # +DM
    mdm = SMMA(ld.where((ld > 0) & (hd < ld), 0), time_period)  # -DM
    return pdm, mdm


def DM(high, low, time_period):
    pdm, mdm = _DM_(high, low, time_period)
    return pdm * time_period, mdm * time_period


# MTR:=EXPMEMA(MAX(MAX(HIGH-LOW,ABS(HIGH-REF(CLOSE,1))),ABS(REF(CLOSE,1)-LOW)),N) 不对
# MTR = SMMA(TR,N)
# +DI = +DM/TR*100
# -DI = -DM/TR*100
def DI(high, low, close, time_period):
    pdm, mdm = _DM_(high, low, time_period)
    tr = SMMA(TR(high, low, close), time_period)
    pdi = pdm / tr * 100  # +DI
    mdi = mdm / tr * 100  # -DI
    return pdi, mdi


# DX = ABS(MDI-PDI)/(MDI+PDI)*100
def DX(high, low, close, time_period):
    pdi, mdi = DI(high, low, close, time_period)
    return ABS(pdi - mdi) / (pdi + mdi) * 100.0  # DX


# ADX = SMMA(DX,N)
def ADX(high, low, close, time_period):
    return SMMA(DX(high, low, close, time_period), time_period)


# ADXR:EXPMEMA(ADX,M) 不对
# ADXR = (ADX+REF(ADX,N))/2
def ADXR(high, low, close, time_period):
    adx = ADX(high, low, close, time_period)
    return (adx + REF(adx, time_period)) / 2


# ROC - Rate of change 变动率指标
# ROC = [(Close - Close n periods ago) / (Close n periods ago)] * 100
def ROC(price, time_period=1):
    return PCT_CHANGE(price, time_period) * 100


# ROCP:Rate of change Percentage: (price-prevPrice)/prevPricee
def ROCP(price, time_period=1):
    return (price - REF(price, time_period)) / REF(price, time_period)


# ROCR:Rate of change ratio: (price/prevPrice)
def ROCR(price, time_period=1):
    return price / REF(price, time_period)


# ROCR100:Rate of change ratio 100 scale: (price/prevPrice)*100
def ROCR100(price, time_period=1):
    return ROCR(price, time_period) * 100


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


# def TR(data, time_period):
#     return EMA(EMA(EMA(data, time_period), time_period), time_period)
def TRIX(price, time_period):
    tr = EMA(EMA(EMA(price, time_period), time_period), time_period)
    return (tr - REF(tr)) / REF(tr) * 100


# Commodity Channel Index (CCI) 顺势指标   算法与talib有出入
# CCI = (Typical Price  -  Time period SMA of TP) / (.015 x  Time period Mean Deviation of TP)
# Typical Price (TP) = (High + Low + Close)/3
# Constant = .015
def CCI(high, low, close, time_period=20):
    tp = TP(high, low, close)
    tp_sma = SMA(tp, time_period)
    tp_std = STD(tp, time_period)
    return (tp - tp_sma) / (.015 * tp_std)


# Typical Price = (High + Low + Close)/3
# Raw Money Flow = Typical Price x Volume
# Money Flow Ratio = (14-period Positive Money Flow)/(14-period Negative Money Flow)
# Money Flow Index = 100 - 100/(1 + Money Flow Ratio)
def MFI(high, low, close, volume, time_period):
    tp = TP(high, low, close)
    raw_money_flow = tp * volume
    pos_money_flow = raw_money_flow.where(tp > REF(tp), 0)
    neg_money_flow = raw_money_flow.where(tp < REF(tp), 0)
    money_flow_ratio = SUM(pos_money_flow, time_period) / SUM(neg_money_flow, time_period)
    return 100 - 100 / (1 + money_flow_ratio)


# Percentage Price Oscillator (PPO) 价格震荡百分比指数 和talib有出入
# Percentage Price Oscillator (PPO): {(12-day EMA - 26-day EMA)/26-day EMA} x 100
# Signal Line: 9-day EMA of PPO
# PPO Histogram: PPO - Signal Line
def PPO(price, fast_period=12, slow_period=26, signal_period=9):
    ppo = (EMA(price, fast_period) - EMA(price, slow_period)) * 100.0 / EMA(price, slow_period)
    ppo_signal = EMA(ppo, signal_period)
    ppo_histogram = ppo - ppo_signal
    return ppo, ppo_signal, ppo_histogram


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


def BIAS(price, time_period=24):
    man = SMA(price, time_period)
    return (price - man) / man * 100

###########################################################################


# On Balance Volume (OBV) 能量潮指标 和talib有偏差
# If the closing price is above the prior close price then:
# Current OBV = Previous OBV + Current Volume
# If the closing price is below the prior close price then:
# Current OBV = Previous OBV  -  Current Volume
# If the closing prices equals the prior close price then:
# Current OBV = Previous OBV (no change)
def OBV(price, volume):
    pnv = volume.where(price > REF(price), -volume)
    return pnv.cumsum()


# (p+2*p1+3*p2+2*p3+p4)/9
def TRIMA(price, time_period):
    return


# StochRSI = (RSI - Lowest Low RSI) / (Highest High RSI - Lowest Low RSI)
def STOCHRSI(data, time_period=14):
    rsi = RSI(data, time_period)
    ll_rsi = LL(rsi, time_period)
    hh_rsi = HH(rsi, time_period)
    return (rsi - ll_rsi) / (hh_rsi - ll_rsi)


def AR(high, low, open, time_period):
    open_sum = SUM(open.time_period)
    high_sum = SUM(high.time_period)
    low_sum = SUM(low.time_period)
    return (high_sum - open_sum) / (open_sum - low_sum)





def ochl2ind(open, close, high, low, volume):
    data = pd.DataFrame()

    # for period in [5, 10, 20, 30, 60, 120]:
    #     data['sma_%d' % period] = SMA(close, period)
    #     data['ema_%d' % period] = EMA(close, period)
    #     data['wma_%d' % period] = WMA(close, period)
    #     data['dema_%d' % period] = DEMA(close, period)
    #     data['smma_%d' % period] = SMMA(close, period)
    #     data['tema_%d' % period] = TEMA(close, period)

    for period in [5, 10, 20, 30]:
        data['min_%d' % period] = MIN(close, period)
        data['max_%d' % period] = MAX(close, period)

    # bb
    upper_band, middle_band, lower_band = BBANDS(close, 5, 2, 2)
    data['upper_band'] = upper_band
    data['middle_band'] = middle_band
    data['lower_band'] = lower_band
    print 'bbands'
    # macd
    macd, macdsignal, macdhist = MACD(close, 12, 26, 9)
    data['macd'] = macd
    data['macd_signal'] = macdsignal
    data['macd_hist'] = macdhist
    print 'macd'
    # kdj
    slow_k, slow_d = STOCH(high, low, close, 9, 3, 3)
    data['slow_k'] = slow_k
    data['slow_d'] = slow_d
    data['slow_j'] = 3 * slow_k - 2 * slow_d
    fast_k, fast_d = STOCHF(high, low, close, 9, 3)
    data['fast_k'] = fast_k
    data['fast_d'] = fast_d
    print 'kdj'

    # dmi
    data['tr'] = TR(high, low, close)
    data['atr'] = ATR(high, low, close, 14)
    # pdm, mdm = DM(high, low, 14)
    # data['pdm'] = pdm
    # data['mdm'] = mdm
    pdi, mdi = DI(high, low, close, 14)
    data['pdi'] = pdi
    data['mdi'] = mdi
    data['adx'] = DX(high, low, close, 6)
    data['adxr'] = ADXR(high, low, close, 6)
    print 'dmi'
    # aroon
    # aroon_up, aroon_down = AROON(high, low, 14)
    # data['aroon_up'] = aroon_up
    # data['aroon_down'] = aroon_down
    # data['aroonosc'] = AROONOSC(high, low, 14)
    # print 'aroon'

    data['emv'] = EMV(high, low, volume, 14)
    data['trix'] = TRIX(close, 14)
    data['cci'] = CCI(high, low, close, 14)
    data['rsi'] = RSI(close, 14)
    data['mfi'] = MFI(high, low, close, volume, 14)
    data['willr'] = WILLR(high, low, close, 14)
    data['willr_89'] = WILLR(high, low, close, 89)
    data['obv'] = OBV(close, volume)
    data['vol_roc'] = ROC(volume)
    data['roc'] = ROC(close, 6)
    # min,max
    data['bias'] = BIAS(close, 24)

    data['c_min'] = close - MIN(close, 10)
    return data
