# -*- coding: utf-8 -*-
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import talib
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np

data = web.DataReader('GOOG', data_source='yahoo', start='9/1/2018', end='12/30/2018')
data = pd.DataFrame(data)

C = data['Adj Close']

rsi = talib.RSI(data['Close'], 2)

# MA_Type: 0=SMA, 1=EMA, 2=WMA, 3=DEMA, 4=TEMA, 5=TRIMA, 6=KAMA, 7=MAMA, 8=T3 (Default=SMA)
SMA = talib.MA(C, 5, matype=0)
EMA = talib.MA(C, 5, matype=1)
WMA = talib.MA(C, 5, matype=2)
DEMA = talib.MA(C, 5, matype=3)
TEMA = talib.MA(C, 5, matype=4)
TRIMA = talib.MA(C, 5, matype=5)
KAMA = talib.MA(C, 5, matype=6)
MAMA = talib.MA(C, 5, matype=7)
T3 = talib.MA(C, 5, matype=8)

data = data.join(pd.Series(SMA, name="SMA"))
data["Direction"] = np.sign(C.pct_change() * 100.0)
data = data.dropna()

forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)
X = data.iloc[:, data.columns != "Direction"]
y = data["Direction"]
forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
        color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()
print data
