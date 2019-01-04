from talib import abstract
import pandas as pd
import Indicators as ind

import pandas_datareader.data as web
import matplotlib.pyplot as plt
import talib
import numpy as np

# print (talib.abstract.WMA)
# print (talib.abstract.EMA)
# print (talib.abstract.SMA)

# print (talib.abstract.BBANDS)


# print(talib.abstract.MACD)
# print(talib.abstract.MACDEXT)
# print(talib.abstract.MACDFIX)

# print(talib.abstract.CCI)
# print(talib.abstract.RSI)


# print (talib.abstract.EOM)

# print (talib.abstract.ATR)

# print (talib.abstract.PLUS_DI)
# print (talib.abstract.PLUS_DM)
# print (talib.abstract.MINUS_DI)
# print (talib.abstract.MINUS_DM)
# print (talib.abstract.DX)
# print (talib.abstract.ADX)
# print (talib.abstract.ADXR)

# print (talib.abstract.WILLR)
# print talib.abstract.TRIX
# print talib.abstract.STOCHRSI
#
# print talib.abstract.AROON
# print talib.abstract.AROONOSC
# print talib.abstract.MFI
# print talib.abstract.PPO
# print talib.abstract.OBV
#
# print talib.get_functions()

df = pd.DataFrame()
df['A'] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
print df['A'].clip_lower(0)
# df['B'] = ['NAN', 3, 6, 10, 15, 21, 28, 36, 45]
#
# df['B'] = df['B'].shift(1) + df['A']
# print df['B']
# print df[['A']].apply(lambda x: x.shift(2))

# for i in df['A'].index:
#     print df['A'].iloc[0:i + 1].sum()
#
# print df.cumsum()

# print df.rolling(3).apply(lambda x: (3 - (3 - pd.Series(x).idxmax() - 1)) / 3 * 100)

data = web.DataReader('GOOG', data_source='yahoo', start='9/1/2018', end='12/30/2018')
data = pd.DataFrame(data)
high, low, close = data['High'], data['Low'], data['Close']
print high.rolling(5).apply(lambda h: pd.Series(h).idxmax(), raw=True)
# print high.rolling(5).apply(lambda x: x.argmax())
