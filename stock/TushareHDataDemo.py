# -*- coding: utf-8 -*-
import tushare as ts
import pandas as pd
from sqlalchemy import create_engine

print(ts.__version__)

engine = create_engine('mysql://root:root@127.0.0.1:3306/Stock?charset=utf8')
stock_basics = ts.get_stock_basics()
print len(stock_basics)
i = 0
for code, row in stock_basics.iterrows():
    name = row['name']
    try:
        h_data = ts.get_h_data(code)
        h_data['code'] = code
        h_data['name'] = name
        h_data.to_sql('h_data', engine, if_exists='append')
        i += 1
        print i, code, name, 'loaded'
    except:
        print i, 'loaderror'
