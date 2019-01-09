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

from sqlalchemy import create_engine
import tushare as ts
import numpy as np

engine = create_engine('mysql://root:root@127.0.0.1:3306/Stock?charset=utf8')
# 存款利率
deposit_rate = ts.get_deposit_rate()
deposit_rate.to_sql('deposit_rate', engine, if_exists='append')
# 贷款利率
loan_rate = ts.get_loan_rate()
loan_rate.to_sql('deposit_rate', engine, if_exists='append')
# 存款准备金率
rrr = ts.get_rrr()
rrr.to_sql('deposit_rate', engine, if_exists='append')
# 货币供应量
money_supply = ts.get_money_supply()
money_supply.to_sql('deposit_rate', engine, if_exists='append')
# 货币供应量(年底余额)
money_supply_bal = ts.get_money_supply_bal()
money_supply_bal.to_sql('deposit_rate', engine, if_exists='append')
# 国内生产总值(年度)
gdp_year = ts.get_gdp_year()
gdp_year.to_sql('deposit_rate', engine, if_exists='append')
# 国内生产总值(季度)
gdp_quarter = ts.get_gdp_quarter()
gdp_quarter.to_sql('deposit_rate', engine, if_exists='append')
# 三大需求对GDP贡献
gdp_for = ts.get_gdp_for()
gdp_for.to_sql('deposit_rate', engine, if_exists='append')
# 三大产业贡献率
gdp_contrib = ts.get_gdp_contrib()
gdp_contrib.to_sql('deposit_rate', engine, if_exists='append')
# 居民消费价格指数
cpi = ts.get_cpi()
cpi.to_sql('deposit_rate', engine, if_exists='append')
# 工业品出厂价格指数
ppi = ts.get_ppi()
ppi.to_sql('deposit_rate', engine, if_exists='append')



