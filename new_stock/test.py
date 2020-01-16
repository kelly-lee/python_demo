import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor

from new_stock import tushares
from new_stock import charts
from new_stock import indicators

base_path = '/Users/kelly.li/stocks/china/tushare/'
#http://www.iwencai.com/stockpick/search?typed=1&preParams=&ts=1&f=1&qs=stockpick_ambiguity_index&selfsectsn=&querytype=stock&searchfilter=&tid=stockpick&multiIndex=%E5%87%80%E8%B5%84%E4%BA%A7%E6%94%B6%E7%9B%8A%E7%8E%87roe&stockpick&w=%E5%87%80%E8%B5%84%E4%BA%A7%E6%94%B6%E7%9B%8A%E7%8E%87roe

def get_daily(ts_code):
    return pd.read_csv(base_path + 'daily/%s.csv' % str(ts_code[0:6]))


def draw(ts_code, start, end):
    stock = get_daily(ts_code)
    stock.rename(columns={'trade_date': 'date'}, inplace=True)
    stock = stock.loc[(stock['date'] > start) & (stock['date'] < end), :].reset_index(drop=True)
    # ax = plt.gca()
    # charts.drawK(ax, stock)
    # charts.drawDate(ax, stock)
    # plt.show()
    charts.drawAll(ts_code, stock, types=[['K', 'SMA'], ['WR'], ['MACD']])
    # plt.plot(stock.close)
    # plt.show()
    # fig = plt.figure(figsize=(16, 8))
    # ax = fig.add_subplot(1, 1, 1)
    # charts.drawK(ax, stock)
    # charts.drawSMA(ax,stock,periods=[20, 60])
    # charts.drawDate(ax, stock)
    # charts.drawWR(ax,stock)
    # plt.show()
    # ax = plt.twinx()


# draw('000860.SZ', 20190601, 201901231)
# 交大昂立

# print('hello')
# stock = get_daily('600276.SH')
# stock.rename(columns={'trade_date': 'date', 'vol': 'volume'}, inplace=True)
# stock['date'] = pd.to_datetime(stock['date'], format='%Y%m%d')
# stock['date_ym'] = stock['date'].map(lambda x: x.strftime('%Y-%m'))
# stock.loc[stock.pct_chg > 9, :].groupby('date_ym').agg({'pct_chg': 'count'}).sort_index()
# print(stock)

# 直方密度图（多图）
def displot_mul(data, feature_xs=None, feature_h=None, grid=(3, 3), bins=10, hist=True, kde=True):
    flg = plt.figure()
    if feature_xs is None:
        feature_xs = data.dtypes[(data.dtypes == 'int64') | (data.dtypes == 'float64')].index.tolist()
    for index, feature_x in enumerate(feature_xs):
        ax = flg.add_subplot(grid[0], grid[1], index + 1)
        displot(data, feature_x=feature_x, feature_h=feature_h, bins=bins, hist=hist, kde=kde)
        ax.set_xlabel(feature_x)
        ax.legend()



# stock_inds = indicators.all_ind(stock)
#
# stock_inds = stock_inds.iloc[-3500:-1, :]
# gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt',
#                                 min_samples_leaf=15, min_samples_split=10, loss='huber', random_state=42)
# y = stock_inds.pop('pct_next')
# X = stock_inds
# gbr.fit(X, y)
# y_pred = gbr.predict(X)
# result = zip(y_pred, y)
# columns = X.columns
# feature_importances = gbr.feature_importances_
# sorted_feature_importances = feature_importances[np.argsort(-feature_importances)]
# feature_importance_names = columns[np.argsort(-feature_importances)]
# for index, name in enumerate(feature_importance_names):
#     print(name, sorted_feature_importances[index])
#
# plt.plot(y_pred)
# plt.plot(y)
# plt.show()
#
# plt.scatter(range(500), y_pred[-500:], s=1)
# plt.scatter(range(500), y[-500:], s=1)
# plt.show()
#
# a = pd.DataFrame({'y_pred': y_pred, 'y': y})
# a['dif'] = abs(a.y_pred - a.y)
# b = a.sort_values(by=['dif'], ascending=False)
# a.loc[(a.y_pred>0) & (a.y<0),:].describe()
