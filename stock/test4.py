import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.dates import MONDAY, DateFormatter, DayLocator, WeekdayLocator
import pandas_datareader.data as web
from mpl_finance import candlestick_ohlc

date1 = "2018-9-1"
date2 = "2019-1-30"

mondays = WeekdayLocator(MONDAY)  # major ticks on the mondays
alldays = DayLocator()  # minor ticks on the days
weekFormatter = DateFormatter('%b %d')  # e.g., Jan 12
dayFormatter = DateFormatter('%d')  # e.g., 12

# quotes = pd.read_csv('data/yahoofinance-INTC-19950101-20040412.csv',
#                      index_col=0,
#                      parse_dates=True,
#                      infer_datetime_format=True)
data = web.DataReader('300059.sz', data_source='yahoo', start='1/9/2018', end='1/30/2019')
quotes = pd.DataFrame(data)

# select desired range of dates
quotes = quotes[(quotes.index >= date1) & (quotes.index <= date2)]

fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.2)
ax.xaxis.set_major_locator(mondays)
ax.xaxis.set_minor_locator(alldays)
ax.xaxis.set_major_formatter(weekFormatter)
# ax.xaxis.set_minor_formatter(dayFormatter)

# plot_day_summary(ax, quotes, ticksize=3)
candlestick_ohlc(ax, zip(mdates.date2num(quotes.index.to_pydatetime()),
                         quotes['Open'], quotes['High'],
                         quotes['Low'], quotes['Close']),
                 width=0.6, colorup='r', colordown='green')

ax.xaxis_date()
ax.autoscale_view()
plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')

plt.show()
