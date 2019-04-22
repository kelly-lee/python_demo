# -*- coding: utf-8 -*-
import baostock as bs
import pandas as pd
import MySQLdb as db
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def format_code(src_code):
    arr = src_code.split(".")
    return arr[1] + '.' + arr[0].upper()


def parse_code(format_code):
    arr = format_code.split(".")
    return arr[1] + '.' + arr[0].lower()


def get_stock_basic():
    bs.login()
    rs = bs.query_stock_basic()
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    data = pd.DataFrame(data_list, columns=rs.fields)
    data.rename(
        columns={'code_name': 'name', 'ipoDate': 'ipo_date', 'outDate': 'out_date'}, inplace=True)
    bs.logout()
    return data


def get_dupont(code):
    dupont_list = []
    for year in range(2000, 2020):
        for quarter in [4]:
            rs_dupont = bs.query_dupont_data(code=code, year=year, quarter=quarter)
            while (rs_dupont.error_code == '0') & rs_dupont.next():
                dupont_list.append(rs_dupont.get_row_data())
    result_profit = pd.DataFrame(dupont_list, columns=rs_dupont.fields)
    return result_profit


def get_a_daily(code, start_date, end_date, frequency):
    code = parse_code(code)
    bs.login()
    rs = bs.query_history_k_data_plus(code,
                                      "date,code,open,high,low,close,volume,turn,tradestatus,pctChg",
                                      start_date=start_date, end_date=end_date,
                                      frequency=frequency, adjustflag="3")
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    data = pd.DataFrame(data_list, columns=rs.fields)
    bs.logout()

    data['code'] = data['code'].apply(lambda code: format_code(code))
    data.rename(
        columns={'pctChg': 'pct_chg', 'tradestatus': 'trade_status'}, inplace=True)
    return data


def save_dupont():
    bs.login()
    stock_basics = get_stock_basic()
    stock_basics = stock_basics[(stock_basics['type'] == '1') & (stock_basics['status'] == '1')]
    data = pd.DataFrame()
    conn = db.connect(host="localhost", port=3306, user="root", password="root", db="stock", charset="utf8")
    i = 0
    for index, stock_basic in stock_basics.iterrows():
        i = i + 1
        # if i > 5:
        #     break
        dupont = get_dupont(stock_basic['code'])
        data = data.append(dupont)
        print('load', index, stock_basic['code'], stock_basic['name'])
    print(data)
    data.to_csv('dupont.csv', index=False)


if __name__ == '__main__':
    basic = get_stock_basic()
    print(basic.info())
    dupont = pd.read_csv('dupont.csv')
    dupont = pd.merge(basic, dupont, on=['code'])

    for code in dupont['code'].unique():
        data = dupont[dupont['code'] == code]
        name = data['name'].unique()[0]
        data = data[(data['statDate'] > '2013-12-31')]
        data.index = data['statDate']
        if len(data) < 5:
            continue
        if (data['dupontROE'].min() < 0):
            continue
        if (data['dupontROE'].max() > 1):
            continue
        if (data['dupontROE'].mean() < 0.23):
            continue
        if (data['dupontROE'].mean() > 0.25):
            continue
        print(name, data['dupontROE'].mean())
        font = FontProperties(fname='/System/Library/Fonts/PingFang.ttc', size=5)

        plt.plot(data['dupontROE'], label=name)

    plt.legend(prop=font)
        # plt.plot()
    plt.show()

    # print(dupont[(dupont['dupontROE'] > 0.2) & (dupont['statDate'] == '2018-12-31')][['name', 'dupontROE']])
