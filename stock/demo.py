# -*- coding: utf-8 -*-
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import talib


# SMA: 10-period sum / 10
def SMA(data, ndays):
    return data.rolling(ndays).mean()


# Initial SMA: 10-period sum / 10
# Multiplier: (2 / (Time periods + 1) ) = (2 / (10 + 1) ) = 0.1818 (18.18%)
# EMA: {Close - EMA(previous day)} x multiplier + EMA(previous day)
def EMA(data, ndays):
    return data.ewm(ndays).mean()


# CCI = (Typical Price  -  Time period SMA of TP) / (.015 x  Time period Mean Deviation of TP)
# Typical Price (TP) = (High + Low + Close)/3
# Constant = .015
def CCI(data, ndays):
    H, L, C = data['High'], data['Low'], data['Close']
    TP = (H + L + C) / 3
    TP_MEAN = SMA(TP, ndays)
    TP_STD = TP.rolling(ndays).std()
    return (TP - TP_MEAN) / (.015 * TP_STD)


# Distance Moved = ((H + L)/2 - (Prior H + Prior L)/2)
# Box Ratio = ((V/100,000,000)/(H - L))
# 1-Period EMV = ((H + L)/2 - (Prior H + Prior L)/2) / ((V/100,000,000)/(H - L))
# 14-Period Ease of Movement = 14-Period simple moving average of 1-period EMV
def EMV(data, ndays):
    H, L, V = data['High'], data['Low'], data['Volume']
    DM = (H + L) / 2 - (H.shift(1) + L.shift(1)) / 2
    BR = ((V / 100000000) / (H - L))
    EMV = DM / BR
    return EMV.rolling(ndays).mean()


# ROC = [(Close - Close n periods ago) / (Close n periods ago)] * 100
def ROC(data, ndays):
    C = data['Close']
    return (C.diff(ndays) - C.shift(ndays)) / C.shift(ndays) * 100


# MACD: (12-day EMA - 26-day EMA)
# Signal Line: 9-day EMA of MACD
# MACD Histogram: MACD - Signal Line
def MACD(data):
    C = data['Close']
    EMA_12 = C.ewm(12).mean()
    EMA_26 = C.ewm(26).mean()
    MACD = EMA_12 - EMA_26
    SL = MACD.ewm(9).mean()
    MACD_H = MACD - SL
    return MACD, SL, MACD_H


# TR : MAX(MAX((HIGH-LOW),ABS(REF(CLOSE,1)-HIGH)),ABS(REF(CLOSE,1)-LOW));
#    ATR : MA(TR,N)
def ATR(data, ndays):
    H, L, C = data['High'], data['Low'], data['Close']
    df = pd.DataFrame()
    df['HL'] = (H - L)
    df['HCL'] = (C.shift(1) - H).abs()
    df['CLL'] = (C.shift(1) - L).abs()
    return df.max(axis=1).rolling(ndays).mean()


data = web.DataReader('GOOG', data_source='yahoo', start='9/1/2018', end='12/30/2018')
data = pd.DataFrame(data)
Close = data['Close']
SMA_100 = SMA(data['Close'], 100)
SMA_50 = SMA(data['Close'], 50)
SMA_10 = SMA(data['Close'], 10)
EWMA_10 = EMA(data['Close'], 10)
CCI_20 = CCI(data, 20)
EMV_14 = EMV(data, 14)
MACD, SL, MACD_H = MACD(data)
ROC_12 = ROC(data, 12)
EMV_14 = EMV(data, 14)
ATR_14 = ATR(data, 14)

fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(2, 1, 1)
ax.set_xticklabels([])
# ax.grid(True)
# ax.plot(Close, lw=1)

ax.plot(SMA_50)
ax.plot(SMA_10)
ax.plot(EWMA_10)

ax = fig.add_subplot(2, 1, 2)
# ax.plot(MACD)
# ax.plot(SL)

# ax.plot(CCI_20)
# ax.fill_between(data.index, 100, CCI_20, where=CCI_20 >= 100, facecolor='green')
# ax.fill_between(data.index, -100, CCI_20, where=CCI_20 <= -100, facecolor='red')

# ax.plot(ROC_12)
# ax.plot(EMV_14)
ax.plot(ATR_14)

ax.grid(True)
plt.setp(plt.gca().get_xticklabels(), rotation=30)
plt.show()

print
rsi = talib.RSI(data['Close'], 2)
print rsi