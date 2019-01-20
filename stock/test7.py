import TushareStore as store
import pandas_datareader.data as web
import MySQLdb as db
import  pandas as pd
import  matplotlib.pyplot as plt
# df = web.DataReader('HMI', start='1/1/2015', end='1/17/2019', data_source='yahoo')
# print df

con = db.connect('localhost', 'root', 'root', 'stock',charset='utf8')
sql = """
select t.*,Name_CN from
(select symbol,max(adj_close),min(adj_close),(max(adj_close)-min(adj_close))/min(adj_close) as roc 
from nasdaq_daily where date>'2015-01-01'
group by symbol )t
inner join usa_company on 
t.symbol = usa_company.symbol 
order by roc desc limit 0,120
"""
stocks = pd.read_sql(sql, con=con)
print stocks['symbol'].tolist()
col = 12
row = 10
fig = plt.figure(figsize=(16,8))
i = 0
for symbol in stocks['symbol'].tolist():
    i+=1
    prices = store.get_usa_daily_data_ind(symbol=symbol)
    # print prices
    ax = fig.add_subplot(row, col, i)
    ax.set_ylabel(symbol)
    ax.plot(prices['adj_close'])
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()
