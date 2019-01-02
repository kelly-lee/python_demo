import talib
from talib import abstract
import pandas as pd

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

print (talib.abstract.ATR)

# print (talib.abstract.PLUS_DI)
# print (talib.abstract.PLUS_DM)
# print (talib.abstract.MINUS_DI)
# print (talib.abstract.MINUS_DM)
# print (talib.abstract.DX)
# print (talib.abstract.ADX)
# print (talib.abstract.ADXR)

# print (talib.abstract.WILLR)
print talib.abstract.TRIX
print talib.abstract.STOCHRSI

print talib.abstract.AROON
print talib.abstract.AROONOSC

print talib.get_functions()

df = pd.DataFrame()
df['A'] = [1, 2, 3, 4, 5, 6, 7, 8, 9]

print df.rolling(3).apply(lambda x: (3 - (3 - pd.Series(x).idxmax() - 1)) / 3 * 100)
