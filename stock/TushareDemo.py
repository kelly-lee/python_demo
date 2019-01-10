# -*- coding: utf-8 -*-
import tushare as ts
import pandas as pd
from sqlalchemy import create_engine

print(ts.__version__)

# print ts.get_industry_classified()
# print ts.get_deposit_rate()
engine = create_engine('mysql://root:root@127.0.0.1:3306/Stock?charset=utf8')
stock_basics = ts.get_stock_basics()
# 概念分类
concept_classified = ts.get_concept_classified()
concept_classified_group = concept_classified.groupby(['code'])['c_name'].apply(list)
concept_classified_group = concept_classified_group.apply(lambda o: repr(o))
print concept_classified_group
stock_basics['ccg'] = concept_classified_group.astype(basestring)
print stock_basics
stock_basics.to_sql('stock_basics', engine, if_exists='append')
# sme_classified = ts.get_sme_classified()
# sme_classified['sme'] = '中小板'
# stock = pd.merge(stock_basics, concept_classified_group, how='left', on=['code','name'])
# print stock.head(100)

