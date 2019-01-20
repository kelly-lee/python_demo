import TushareStore as store
import pandas_datareader.data as web
import MySQLdb as db
import  pandas as pd
import  matplotlib.pyplot as plt
# df = web.DataReader('HMI', start='1/1/2015', end='1/17/2019', data_source='yahoo')
# print df



con = db.connect('localhost', 'root', 'root', 'stock',charset='utf8')
sql = """
select t1.symbol as symbol from usa_company_4 as t1
left join usa_company as t2 
on t1.symbol = t2.symbol where t2.symbol is null
"""
delete_sql = """
    delete from usa_core_daily where symbol  =  '%s'
"""
stocks = pd.read_sql(sql, con=con)
print stocks['symbol'].tolist()
# col = 12
# row = 10
# fig = plt.figure(figsize=(16,8))
i = 0
for symbol in stocks['symbol'].tolist():
    i+=1
    cursor = con.cursor()
    print delete_sql % (symbol)
    cursor.execute(delete_sql % (symbol))
    con.commit()
    print 'delete',symbol
    # prices = store.get_usa_daily_data_ind(symbol=symbol)
    # # print prices
    # ax = fig.add_subplot(row, col, i)
    # ax.set_ylabel(symbol)
    # ax.plot(prices['adj_close'])
    # ax.set_xticks([])
    # ax.set_yticks([])


# plt.show()
