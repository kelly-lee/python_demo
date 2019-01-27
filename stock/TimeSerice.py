# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # seaborn画出的图更好看，且代码更简单，缺点是可塑性差
from statsmodels.graphics.tsaplots import plot_acf  # 自相关图
from statsmodels.tsa.stattools import adfuller as ADF  # 平稳性检测
from statsmodels.graphics.tsaplots import plot_pacf  # 偏自相关图
from statsmodels.stats.diagnostic import acorr_ljungbox  # 白噪声检验
from statsmodels.tsa.arima_model import ARIMA
# seaborn 是建立在matplotlib之上的
import pandas_datareader.data as web

# %matplotlib inline
# %pylab inline

# jupyter中文显示是方框，加入下面两行即可显示中文，若嫌麻烦，也可去网上搜索如何永久显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# pylab.rcParams['figure.figsize'] = (10, 6)   #设置输出图片大小
sns.set(color_codes=True)  # seaborn设置背景

# 读取数据，指定日期列为指标，Pandas自动将“日期”列识别为Datetime格式
# data = pd.read_excel('arima_data.xls', index_col = u'日期')
data = web.DataReader('amzn', data_source='yahoo', start='2018-06-01', end='2019-01-26')
close = data['Close']
# close.plot()
# 自相关系数长期大于零，说明时间序列有很强的相关性
# plot_acf(close)
# plt.show()

diff = close.diff().dropna()
print diff
# diff.plot()
# plt.show()
# 单位根检验p值小于0.05，所以1阶差分后的序列是平稳序列
# print(u'差分序列的白噪声检验结果为：', acorr_ljungbox(diff, lags=1))  # 返回统计量和p值
plot_acf(diff)
plt.show()

# 返回值依次为adf、pvalue、usedlag、nobs、critical values、icbest、regresults、resstore
# (u'\u539f\u59cb\u5e8f\u5217\u7684ADF\u68c0\u9a8c\u7ed3\u679c\u4e3a\uff1a',
# (-1.274877130004426, 0.6405862638158206, 11, 1012,
# {'5%': -2.8644002004847144, '1%': -3.436828225807217, '10%': -2.568292900881126}, 7981.264737261272))
# print(u'原始序列的ADF检验结果为：', ADF(close))
# diff_close = close.diff().dropna()  # 1阶差分，丢弃na值
# plt.plot(diff_close)
# plt.show()
