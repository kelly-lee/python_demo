import tushare as ts
import pandas as pd
base_path = '/Users/kelly.li/stocks/china/tushare/'
ts.set_token('4a988cfe3f2411b967592bde8d6e0ecbee9e364b693b505934401ea7')
pro = ts.pro_api()


def load_stock_basic(pro):
    data = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
    data.to_csv('/Users/kelly.li/stocks/china/tushare/stock_basic.csv', index=False)


# 初始化执行
def load_all_daily():
    stock_basic = pd.read_csv(base_path + 'stock_basic.csv')
    for index, row in stock_basic.iterrows():
        try:
            stock = ts.pro_bar(ts_code=row.ts_code, adj='qfq', adjfactor=True)
            stock = stock.sort_values(by=['trade_date'], ascending=True).reset_index(drop=True)
            stock.to_csv(base_path + 'daily/%s.csv' % str(row.ts_code[0:6]), index=False)
        except Exception:
            print('load error %s' % row.ts_code)




# 每天执行
# def load_daily(start_date, end_date):
#     stock_basic = pd.read_csv(base_path + 'stock_basic.csv')
#     for index, row in stock_basic.iterrows():
#         try:
#             stock_daily_path = base_path + 'daily/%s.csv' % str(row.ts_code[0:6])
#             stock = pd.read_csv(stock_daily_path)
#             daily_stock = ts.pro_bar(ts_code=row.ts_code, start_date=start_date, end_date=end_date, adj='qfq', adjfactor=True)
#             daily_stock = daily_stock.sort_values(by=['trade_date'], ascending=True).reset_index(drop=True)
#             stock = stock.append(daily_stock)
#             stock.to_csv(stock_daily_path, index=False)
#             print('load %s' % row.ts_code)
#         except Exception:
#             print('load error %s' % row.ts_code)


def load_daily(start_date, end_date):
    stock_basic = pd.read_csv(base_path + 'stock_basic.csv')
    error_list = []
    for index, row in stock_basic.iterrows():
        try:
            stock_daily_path = base_path + 'daily/%s.csv' % str(row.ts_code[0:6])
            stock = pd.read_csv(stock_daily_path)
            daily_stock = ts.pro_bar(ts_code=row.ts_code, start_date=start_date, end_date=end_date, adj='qfq', adjfactor=True)
            daily_stock = daily_stock.sort_values(by=['trade_date'], ascending=True).reset_index(drop=True)
            stock = stock.append(daily_stock)
            stock = stock.drop_duplicates()
            stock.to_csv(stock_daily_path, index=False)
        except Exception:
            error_list.append(row.ts_code)
            print('load error %s' % row.ts_code)
    return error_list


def load_error_daily(error_list, start_date, end_date):
    new_error_list = []
    for row in error_list:
        # try:
        stock_daily_path = base_path + 'daily/%s.csv' % str(row[0:6])
        print(row, stock_daily_path)
        stock = pd.read_csv(stock_daily_path)
        daily_stock = ts.pro_bar(ts_code=row, start_date=start_date, end_date=end_date, adj='qfq', adjfactor=True)
        print(daily_stock)
        daily_stock = daily_stock.sort_values(by=['trade_date'], ascending=True).reset_index(drop=True)
        stock = stock.append(daily_stock)
        stock = stock.drop_duplicates()
        stock.to_csv(stock_daily_path, index=False)
        # except Exception:
        #     new_error_list.append(row)
        #     print('load error %s' % row)
    return new_error_list


def task():
    error_list = load_daily(start_date='20191224', end_date='20191225')
    new_error_list = load_error_daily(error_list=error_list, start_date='20191224', end_date='20191225')

def testABC():
    print('test')
# start_date = '20191227'
# end_date = '20200115'
# error_list = load_daily(start_date=start_date, end_date=end_date)
# new_error_list = load_error_daily(error_list=error_list, start_date=start_date, end_date=end_date)


# data = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
# data.to_csv('/Users/kelly.li/stocks/china/tushare/stock_basic.csv',index=False)
# bj = data.loc[data['industry']=='白酒',:]
# stock = pd.read_csv(base_path + 'daily/%s.csv' % str('600276'))
# daily_stock = ts.pro_bar(ts_code='600276.SH',start_date='20190124',end_date = '20190125', adj='qfq', adjfactor=True)
# daily_stock = daily_stock.sort_values(by=['trade_date'], ascending=True).reset_index(drop=True)
# stock = stock.append(daily_stock)



