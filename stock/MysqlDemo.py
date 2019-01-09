# -*- coding: utf-8 -*-
import MySQLdb as mysql
import pandas as pd
import tushare as ts

# hs300 = ts.get_hs300s()
# print hs300
#
# try:
#     conn = mysql.connect(user='root', passwd='root')  # cennect the database
#     cur = conn.cursor()  # get the cur
#     cur.execute('create database if not exists Stock')
#     conn.select_db('Stock')
#     cur.execute('create table if not exists hs300(code varchar(10),weight integer)')
#     hs300 = ts.get_hs300s()  # get all the data and other data will be add to this d
#     for cnt in range(0, len(hs300)):  # 将hs300中的数据存储到数据库中
#         SQL = 'INSERT INTO hs300 (code,weight) VALUES (%s,%s)' % (hs300['code'][cnt], float(hs300['weight'][cnt]))
#         cur.execute(SQL)
#     conn.commit()  # 执行上诉操作
#     cur.close()
#     conn.close()
#
# except mysql.Error, e:
#     print "Mysql Error %d: %s" % (e.args[0], e.args[1])
#
# conn = mysql.connect(host='localhost', user='root', passwd='root', db='test', charset='utf8')
# print conn
#
# sql = "select * from user limit 3"
# df = pd.read_sql(sql, conn, index_col="id")
# print df


from sqlalchemy import create_engine
import tushare as ts
import numpy as np

engine = create_engine('mysql://root:root@127.0.0.1:3306/Stock?charset=utf8')

# df = ts.get_hist_data('600848')
# print df
# # engine.execute('CREATE INDEX ix_d_data_date ON hist_data (date(20))')
# # 存入数据库
# df.to_sql('hist_data', engine, if_exists='append')

# stock_basics = ts.get_stock_basics()
# engine.execute('CREATE INDEX ix_stock_basics_code ON stock_basics (date(20))')
# stock_basics.to_sql('stock_basics', engine, if_exists='append')

report_data = ts.get_report_data(2018, 3)
report_data['year'] = 2018
report_data['season'] = 3
report_data.to_sql('report_data', engine, if_exists='append')
print report_data

# for year in range(2000, 2019):
#     for season in [1, 2, 3, 4]:
#         report_data = ts.get_report_data(year, season)
#         report_data['year'] = year
#         report_data['season'] = season
#         try:
#             report_data.to_sql('report_data', engine, if_exists='append')
#         except IOError, e:
#             print "Mysql Error %d: %s" % (e.args[0], e.args[1])

