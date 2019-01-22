# -*- coding: utf-8 -*-
import urllib2
# import urllib.request
import json
import pandas as pd


def get_xuqiu_socket_list(page):
    url = "https://xueqiu.com/stock/cata/stocklist.json?page=" + str(
        page) + "&size=90&order=desc&orderby=marketCapital&type=0%2C1%2C2%2C3&isdelay=1"
    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'zh-CN,zh;q=0.9',
        'Connection': 'keep-alive',
        'Cookie': 'aliyungf_tc=AQAAAALwsQp3fw4ABrb+ZXMGGSlpc9To; _ga=GA1.2.1224636926.1544524176; device_id=d26ab5c8ca82876fbd126befe09f41f2; xq_a_token.sig=6vYUkYxPfp54csZymplW5bCPODE; xq_r_token.sig=aKD-9IKtPgyogiXsU8XCdFMevnY; u=351545898078432; s=ea1178np0c; __utmc=1; __utmz=1.1547800722.1.1.utmcsr=baidu|utmccn=(organic)|utmcmd=organic; xq_a_token=dac65245b3a3efae1b7df05a0da1e391a1dc9135; xq_r_token=24a12835d176d574c10d976cfc672b9a9d73eba7; __utma=1.1224636926.1544524176.1548122875.1548125953.3; Hm_lvt_1db88642e346389874251b5a1eded6e3=1545898077,1547800723,1548133220; Hm_lpvt_1db88642e346389874251b5a1eded6e3=1548133220; _gid=GA1.2.533413683.1548133223',
        'Host': 'xueqiu.com',
        'Control': 'max-age=0',
        'Upgrade-Insecure-Requests': '1',
        # 'Referer':' https://xueqiu.com/',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.110 Safari/537.36',
        # 'X-Requested-With':' XMLHttpRequest'
    }

    req = urllib2.Request(url, headers=headers)
    response = urllib2.urlopen(req)
    html = response.read()
    stocks = json.loads(html)['stocks']
    data = [[stock['symbol'], stock['name'], stock['marketcapital'], stock['pettm']] for stock in stocks]
    df = pd.DataFrame(data, columns=['symbol', 'name', 'marketcapital', 'pettm'])
    return df


def save_xuqiu_socket_list():
    df = pd.DataFrame()
    for i in range(1, 137):
        print i
        df = df.append(get_xuqiu_socket_list(i))
    df.to_csv('CompanyList_Xuqiu_CN.csv', header=True, index=False, encoding='utf-8')


list_cn = pd.read_csv('CompanyList_CN.csv')
list = pd.read_csv('CompanyList.csv')
print list
list = list[list["sector"].notnull()]
print len(list)

list_merge = pd.merge(list_cn, list, how='inner', on=['symbol'])
list_merge.to_csv('CompanyList_All.csv', header=True, index=False, encoding='utf-8')




