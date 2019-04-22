# -*- coding: utf-8 -*-
# from __future__ import print_function
from sqlalchemy import create_engine

import sys
import pandas_datareader.data as web
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import pandas as pd
import MySQLdb as db
# from sklearn import cluster, covariance, manifold
# from bs4 import BeautifulSoup
# import urllib2
from bs4 import UnicodeDammit
import re
import sys
import socket.Charts as Charts

import socket.Indicators as ind


# 获得美股列表
def get_usa_company_list(exchange, sector):
    company_list = pd.read_csv('CompanyList.csv')
    if (len(exchange) > 0) & (not exchange.isspace()):
        company_list = company_list[company_list.exchange == exchange]
    if (len(sector) > 0) & (not sector.isspace()):
        company_list = company_list[company_list.sector == sector]
    return company_list


# 合并并存储美股列表
def merge_and_save_usa_company_list():
    company = pd.DataFrame()
    amex = pd.read_csv('CompanyList_AMEX.csv')
    amex.drop(["LastSale", "Summary Quote"], inplace=True, axis=1)
    amex.rename(
        columns={"Symbol": "symbol", "Name": "name", "MarketCap": "market_cap", "IPOyear": "ipo_year",
                 "Sector": "sector",
                 "Industry": "industry"}, inplace=True)
    amex.drop(amex.columns[-1], inplace=True, axis=1)
    amex['exchange'] = 'AMEX'
    nasdaq = pd.read_csv('CompanyList_NASDAQ.csv')
    nasdaq.drop(["LastSale", "Summary Quote", "ADR TSO"], inplace=True, axis=1)
    nasdaq.drop(nasdaq.columns[- 1], inplace=True, axis=1)
    nasdaq.rename(
        columns={"Symbol": "symbol", "Name": "name", "MarketCap": "market_cap", "IPOyear": "ipo_year",
                 "Sector": "sector",
                 "Industry": "industry"}, inplace=True)
    nasdaq['exchange'] = 'NASDAQ'
    nyse = pd.read_csv('CompanyList_NYSE.csv')
    nyse.drop(["LastSale", "Summary Quote"], inplace=True, axis=1)
    nyse.rename(
        columns={"Symbol": "symbol", "Name": "name", "MarketCap": "market_cap", "IPOyear": "ipo_year",
                 "Sector": "sector",
                 "Industry": "industry"}, inplace=True)
    nyse.drop(nyse.columns[- 1], inplace=True, axis=1)
    nyse['exchange'] = 'NYSE'
    company = company.append(amex)
    company = company.append(nasdaq)
    company = company.append(nyse)
    company.apply(lambda x: str(x).isspace())
    company['sector'].replace(
        ['Capital Goods', 'Health Care', 'Energy', 'Technology', 'Basic Industries', 'Finance', 'Consumer Services',
         'Public Utilities', 'Miscellaneous', 'Consumer Durables', 'Transportation', 'Consumer Non-Durables'],
        ['capital_goods', 'health_care', 'energy', 'technology', 'basic_industries', 'finance', 'consumer_services',
         'public_utilities', 'miscellaneous', 'consumer_durables', 'transportation', 'consumer_non_durables'],
        inplace=True)
    company.to_csv('CompanyList.csv', index=False, header=True, encoding='utf-8')


# 根据股票代码获得和纳斯达克指数趋势相关性
def get_cor_with_ixic(sector, symbols, start, end):
    df = pd.DataFrame()
    nasdaq = web.DataReader('^IXIC', start=start, end=end, data_source='yahoo')
    nasdaq['date'] = nasdaq.index.to_period("D")
    nasdaq = nasdaq.iloc[1:]
    for symbol in symbols:
        prices = get_usa_daily_data_ind(sector, symbol=symbol, start_date=start, end_date=end)
        # nasdaq_match = pd.merge(prices.date.to_frame(), nasdaq, how='left', on=['date'])
        # print nasdaq.dtypes, prices.dtypes
        # print nasdaq_match
        # print len(prices), len(nasdaq_match)
        if len(prices) != len(nasdaq):
            continue
        cor = np.corrcoef(nasdaq.Close.tolist(), prices.close.tolist())[0, 1]
        df.at[symbol, 'cor'] = cor
        df.at[symbol, 'symbol'] = symbol
    df.sort_values(by=['cor'], ascending=False, inplace=True)
    df.to_csv('CompanyList_cor.csv', index=False, header=True, encoding='utf-8')
    return df


def get_company_by_price_drct(drt):
    # 查询每个股票最高价出现的时间
    sql_max = """
    select t1.symbol as symbol,t1.adj_close as max_close,unix_timestamp(t1.date) as max_ts from usa_technology_daily as t1
    inner join (select symbol ,max(adj_close) as max_adj_close from usa_technology_daily where adj_close >1 group by symbol  )t2
    on t1.symbol = t2.symbol and t1.adj_close = t2.max_adj_close
    """
    # 查询每个股票最低价出现的时间
    sql_min = """
    select t1.symbol as symbol,t1.adj_close as min_close,unix_timestamp(t1.date) as min_ts from usa_technology_daily as t1
    inner join (select symbol ,min(adj_close) as min_adj_close from usa_technology_daily where adj_close >1  group by symbol  )t2
    on t1.symbol = t2.symbol and t1.adj_close = t2.min_adj_close
    """

    pf_max = store.query_by_sql(sql_max)
    pf_max = pf_max.drop_duplicates(['symbol'])
    pf_min = store.query_by_sql(sql_min)
    pf_min = pf_min.drop_duplicates(['symbol'])
    df = pd.merge(pf_max, pf_min, how='inner', on=['symbol'])
    df['drt'] = df.max_ts - df.min_ts
    df['roc'] = (df.max_close - df.min_close) / df.min_close
    # df = df.drop_duplicates(['symbol'])
    df.sort_values(by=['roc'], ascending=[0], inplace=True)
    if drt > 0:
        return df[df.drt > 0]
    else:
        return df[df.drt < 0]


# 保存日k线
def save_usa_daily_data(engine, table, symbol, start, end):
    nasdaq_daily = web.DataReader(symbol, start=start, end=end, data_source='yahoo')
    nasdaq_daily.index = nasdaq_daily.index.to_period("D")
    nasdaq_daily['symbol'] = symbol
    nasdaq_daily.rename(
        columns={'Open': 'open', 'Close': 'close', 'High': 'high', 'Low': 'low', 'Volume': 'volume',
                 'Adj Close': 'adj_close'}, inplace=True)
    nasdaq_daily.to_sql('usa_' + table + '_daily', engine, if_exists='append')


# 批量保存日k线
def batch_save_usa_daily_data(sector, symbols, start, end):
    if symbols is None:
        symbols = get_usa_company_list(exchange='', sector=sector)['symbol'].tolist()
    engine = create_engine('mysql://root:root@127.0.0.1:3306/Stock?charset=utf8')
    error_codes = []
    i = 0
    for symbol in symbols:
        i += 1
        try:
            save_usa_daily_data(engine=engine, table=sector, symbol=symbol, start=start, end=end)
            print (i, symbol, 'save')
        except:
            print (i, symbol, 'save error')
            error_codes.append(symbol)
    print (error_codes)


# 查询美股日行情
def get_usa_daily_data_ind(sector='', symbol='', trade_date='', start_date='', end_date='', append_ind=False):
    con = db.connect('localhost', 'root', 'root', 'stock')
    df = pd.DataFrame()
    sql = "SELECT symbol,date,open,close,adj_close,high,low,volume FROM usa_" + sector + "_daily where 1=1 "
    if (len(symbol) > 0) & (not symbol.isspace()):
        sql += "and symbol = %(symbol)s "
    if (len(trade_date) > 0) & (not trade_date.isspace()):
        sql += "and date = %(date)s "
    if (len(start_date) > 0) & (not start_date.isspace()):
        sql += "and date >= %(start_date)s "
    if (len(end_date) > 0) & (not end_date.isspace()):
        sql += "and date <= %(end_date)s "
    sql += "order by symbol asc , date asc "
    print (sql)
    data = pd.read_sql(sql, params={'symbol': symbol, 'date': trade_date, 'start_date': start_date,
                                    'end_date': end_date}, con=con)
    if append_ind:
        open, close, high, low, volume = data['open'], data['close'], data['high'], data['low'], data['volume']
        ochl2ind = ind.ochl2ind(open, close, high, low, volume)
        data = data.join(ochl2ind, how='left')
    df = df.append(data)
    con.close()
    return df


'capital_goods', 'health_care', 'energy', 'technology', 'basic_industries', 'finance', 'consumer_services',
'public_utilities', 'miscellaneous', 'consumer_durables', 'transportation'


def test_batch_save_usa_daily_data():
    start = '2019-01-21'
    end = '2019-01-31'

    # Finance 金融 1054
    # batch_save_usa_daily_data(sector='finance', symbols=None, start=start, end=end)
    # batch_save_usa_daily_data(sector='finance', symbols=['LTSK', 'AGFSW', 'ALACW', 'ALGRW', 'AMBCW', 'AMCIW', 'ARYAW', 'BRPAW', 'BRACR', 'BRACU', 'BRACW', 'BWMCW', 'CTACW',
    #  'BBCPW', 'DDMXW', 'DFBHW', 'EDTXW', 'FTACW', 'FMCIW', 'GPAQW', 'GRSHW', 'GLACW', 'HYACU', 'HYACW', 'HCCHW',
    #  'IEAWW', 'KBLMU', 'KBLMW', 'LACQW', 'LFACW', 'LMFAW', 'LOACW', 'MMDMW', 'MUDSW', 'NHLDW', 'NEBUW', 'OPESW',
    #  'OXBRW', 'WRLSW', 'PACQW', 'SCACW', 'SAMAW', 'STNLW', 'TZACW', 'TCBIW', 'TBRGW', 'TIBRW', 'TKKSW', 'TOTAW',
    #  'TDACW', 'TMCXW', 'TWLVW', 'VEACU', 'VEACW', 'VTIQW', 'ZIONW', 'BAC^W', 'BAC^Y', 'COF^C', 'CIT', 'C^K', 'C^N',
    #  'JPM^A', 'JPM^H', 'OFG^A', 'STL^A', 'SF^A', 'USB^A'], start=start, end=end)

    # Health Care 卫生保健 882
    # batch_save_usa_daily_data(sector='health_care', symbols=None, start=start, end=end)
    batch_save_usa_daily_data(sector='health_care',
                              symbols=['ABEOW', 'BNTCW', 'BVXVW', 'BCACU', 'BCACW', 'CLRBW', 'CLRBZ', 'CHEKW', 'CTXRW',
                                       'CNACW', 'CYTXZ', 'DRIOW', 'NDRAW', 'ENTXW', 'EYEGW', 'HJLIW', 'KANG', 'IMRNW',
                                       'KTOVW', 'MDGSW', 'MTFBW', 'NUROW', 'ONTXW', 'OPGNW', 'PAVMW', 'PAVMZ', 'PHIOW',
                                       'EYESW', 'SRTSW', 'SNGXW', 'VKTXW']
                              , start=start, end=end)

    # 能源 300
    # batch_save_usa_daily_data(sector='energy', symbols=None,
    #                           start=start, end=end)
    # batch_save_usa_daily_data(sector='energy',
    #                           symbols=['AMRWW', 'FLMNW', 'NESRW', 'ROSEW', 'USWSW', 'ZNWAA', 'CPE^A', 'REN']
    #                           , start=start, end=end)

    # 资本货物 377
    # batch_save_usa_daily_data(sector='capital_goods', symbols=None,
    #                           start=start, end=end)
    # batch_save_usa_daily_data(sector='capital_goods',
    #                           symbols=['BNGOW', 'CETXW', 'CLIRW', 'SOLOW', 'TBLTW', 'NAV^D']
    #                           , start=start, end=end)
    # 非耐用消费品 236
    # batch_save_usa_daily_data(sector='consumer_non_durables', symbols=None,
    #                           start=start, end=end)
    # batch_save_usa_daily_data(sector='consumer_non_durables',
    #                           symbols=['TWNKW', 'MKC.V']
    #                           , start=start, end=end)
    # 消费服务 822
    # batch_save_usa_daily_data(sector='consumer_services', symbols=None,
    #                           start=start, end=end)
    # batch_save_usa_daily_data(sector='consumer_services',
    #                           symbols=['APDNW', 'DSKEW', 'TACOW', 'DLPNW', 'LINDW', 'NXTDW', 'EAGLW',
    #                                    'RBZAW', 'SNHNL', 'WINRW', 'WHLRW', 'PSA^B',
    #                                    'PSA^C', 'PSA^D', 'PSA^E', 'PSA^F', 'PSA^U', 'PSA^V', 'TY^']
    #                           , start=start, end=end)
    # 基础工业 316
    # batch_save_usa_daily_data(sector='basic_industries', symbols=None,
    #                           start=start, end=end)
    # batch_save_usa_daily_data(sector='basic_industries', symbols=['SAND', 'ATISW', 'DD^A', 'DD^B', 'HL^B', 'LEN.B'],
    #                           start=start, end=end)
    # 耐用消费品 153
    # batch_save_usa_daily_data(sector='consumer_durables', symbols=None,
    #                           start=start, end=end)
    # batch_save_usa_daily_data(sector='consumer_durables',
    #                           start=start, end=end,
    #                           symbols=['BLNKW', 'CMSSR', 'CMSSU', 'CMSSW', 'JASNW', 'NXEOW', 'RVLT', 'SGLBW'])
    # 运输 124
    # batch_save_usa_daily_data(sector='transportation', symbols=None, start=start,
    #                           end=end)
    # batch_save_usa_daily_data(sector='transportation', start=start, end=end,
    #                           symbols=['CYRXW', 'HUNTU', 'HUNTW', 'SHIPW', 'KSU^', 'SBBC'])
    # 杂 154
    # batch_save_usa_daily_data(sector='miscellaneous', symbols=None, start=start,
    #                           end=end)
    # batch_save_usa_daily_data(start=start, end=end, sector='miscellaneous',
    #                           symbols=['PRTHW','SPB'])
    # 公共设施 270
    # batch_save_usa_daily_data(sector='public_utilities', symbols=None,start='start,end=end)
    # batch_save_usa_daily_data(start=start, end='2019-01-21', sector='public_utilities',
    #                           symbols=['ESTRW', 'JSYNR', 'JSYNU', 'JSYNW', 'AMOV', 'CMS^B', 'DUKB', 'EP^C', 'NMK^B',
    #                                    'NMK^C', 'NI^B', 'SRE^A'])
    # 科技 624
    # batch_save_usa_daily_data(sector='technology', symbols=None,start=start, end=end)
    # batch_save_usa_daily_data( start=start, end=end, sector='technology',
    #                           symbols=['AMRHW', 'CREXW', 'FPAYW', 'GFNSL', 'GTYHW', 'MTECW', 'PHUNW', 'EGHT'])


# Health Care 卫生保健 882|Finance 金融 1054|


def draw_cor(sector, start, end):
    symbols = get_usa_company_list(exchange='', sector=sector)['symbol'].tolist()
    df = get_cor_with_ixic(sector, symbols, start, end)
    cn = pd.read_csv('CompanyList_CN.csv')
    df_cn = pd.merge(df, cn, how='inner', on=['symbol'])
    df_cn.to_csv('CompanyList_' + sector + '_Cor.csv', index=False, header=True, encoding='utf-8')
    df = df.iloc[0: 120]
    Charts.drawPanel(12, 10, sector, df.index.tolist(), start, end)


if __name__ == '__main__':
    # df = get_usa_company_list(sector='finance', exchange='')
    # print df
    # df = df.groupby('sector').count()
    # print df
    test_batch_save_usa_daily_data()

    # merge_and_save_usa_company_list()
    # draw_cor('technology', '2015-01-01', '2019-01-21')
