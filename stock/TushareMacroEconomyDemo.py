# -*- coding: utf-8 -*-
import MySQLdb as mysql
from sqlalchemy import create_engine
import tushare as ts
import matplotlib.pyplot as plt
import pandas as pd


print mysql.__version__

# 指定默认字体
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['font.family'] = 'sans-serif'
# 解决负号'-'显示为方块的问题
# plt.rcParams['axes.unicode_minus'] = False

engine = create_engine('mysql://root:root@127.0.0.1:3306/Stock?charset=utf8')
# # 存款利率
# deposit_rate = ts.get_deposit_rate()
# deposit_rate.to_sql('deposit_rate', engine, if_exists='append')
# # 贷款利率
# loan_rate = ts.get_loan_rate()
# loan_rate.to_sql('loan_rate', engine, if_exists='append')
# # 存款准备金率
# rrr = ts.get_rrr()
# rrr.to_sql('rrr', engine, if_exists='append')
# # 货币供应量
# money_supply = ts.get_money_supply()
# money_supply.to_sql('money_supply', engine, if_exists='append')
# # 货币供应量(年底余额)
# money_supply_bal = ts.get_money_supply_bal()
# money_supply_bal.to_sql('money_supply_bal', engine, if_exists='append')
# # 国内生产总值(年度)
# gdp_year = ts.get_gdp_year()
# gdp_year.to_sql('gdp_year', engine, if_exists='append')
# # 国内生产总值(季度)
# gdp_quarter = ts.get_gdp_quarter()
# gdp_quarter.to_sql('gdp_quarter', engine, if_exists='append')
# # 三大需求对GDP贡献
# gdp_for = ts.get_gdp_for()
# gdp_for.to_sql('gdp_for', engine, if_exists='append')
# # 三大产业贡献率
# gdp_contrib = ts.get_gdp_contrib()
# gdp_contrib.to_sql('gdp_contrib', engine, if_exists='append')
# # 居民消费价格指数
# cpi = ts.get_cpi()
# cpi.to_sql('cpi', engine, if_exists='append')
# # 工业品出厂价格指数
# ppi = ts.get_ppi()
# ppi.to_sql('ppi', engine, if_exists='append')


df = pd.read_sql('ppi', engine)
# df.index = df['month']
df = df.dropna()
df = df.head(50)
df.sort_values(by=['index'], ascending=[0], inplace=True)
print df
# ppiip :工业品出厂价格指数
# ppi :生产资料价格指数
# qm:采掘工业价格指数
# rmi:原材料工业价格指数
# pi:加工工业价格指数
# cg:生活资料价格指数
# food:食品类价格指数
# clothing:衣着类价格指数
# roeu:一般日用品价格指数
# dcg:耐用消费品价格指数

plt.plot(df.month, df.ppiip, label=u'ppiip')
plt.plot(df.month, df.ppi, label=u'ppi')
plt.plot(df.month, df.qm, label=u'qm')
plt.plot(df.month, df.rmi, label=u'rmi')
plt.plot(df.month, df.pi, label=u'pi')
plt.plot(df.month, df.cg, label=u'cg')
plt.plot(df.month, df.food, label=u'food')
plt.plot(df.month, df.clothing, label=u'clothing')
plt.plot(df.month, df.roeu, label=u'roeu')
plt.plot(df.month, df.dcg, label=u'dcg')
plt.setp(plt.gca().get_xticklabels(), rotation=30)
# plt.plot(df)
plt.legend()
plt.show()
