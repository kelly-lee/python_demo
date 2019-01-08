# -*- coding: utf-8 -*-
from talib import abstract
import pandas as pd

import pandas_datareader.data as web
import matplotlib.pyplot as plt
import talib

df = pd.DataFrame()
df['a'] = [-5, -4, -3, -2, -1, 0, 0, 1, 2, 3, 4, 5]
print df.a.abs()
print df.a.clip_lower(0)
df.plot()

