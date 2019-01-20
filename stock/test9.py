import TushareStore as store
import pandas_datareader.data as web
import MySQLdb as db
import pandas as pd
import matplotlib.pyplot as plt

sql = """
select t1.symbol ,t1.adj_close,t1.date as date from usa_core_daily as t1
inner join (select symbol ,max(adj_close) as adj_close from usa_core_daily group by symbol  )t2
on t1.symbol = t2.symbol and t1.adj_close = t2.adj_close
order by date  desc limit 240,120
"""

con = db.connect('localhost', 'root', 'root', 'stock', charset='utf8')
stocks = pd.read_sql(sql, con=con)
# print df

# plt.scatter(df.index, df['date'],s=0.1)
# # plt.show()


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
