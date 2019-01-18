# -*- coding: utf-8 -*-
# from __future__ import print_function

# Author: Gael Varoquaux gael.varoquaux@normalesup.org
# License: BSD 3 clause

import sys
import pandas_datareader.data as web
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import pandas as pd

from sklearn import cluster, covariance, manifold
from bs4 import BeautifulSoup
import urllib2
from bs4 import UnicodeDammit
import re
import TushareStore as store

import sys

reload(sys)
sys.setdefaultencoding('utf8')
type = sys.getfilesystemencoding()
url = "http://vip.stock.finance.sina.com.cn/usstock/ustotal.php"
# 浏览器头
headers = {'User-Agent': 'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US; rv:1.9.1.6) Gecko/20091201 Firefox/3.5.6'}
req = urllib2.Request(url=url, headers=headers)
html = urllib2.urlopen(req).read()
html = unicode(html, 'GBK').encode('UTF-8')
dammit = UnicodeDammit(html)
html = dammit.unicode_markup
print dammit.original_encoding
# print html
html = """
<div class="col_div">
<a href="http://stock.finance.sina.com.cn/usstock/quotes/SINA.html"
	rel="suggest" title="SINA,SINA Corp.,新浪">新浪(SINA)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/SOHU.html"
	rel="suggest" title="SOHU,Sohu.com Ltd.,搜狐">搜狐(SOHU)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/NTES.html"
	rel="suggest" title="NTES,NetEase, Inc.,网易">网易(NTES)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/BIDU.html"
	rel="suggest" title="BIDU,Baidu, Inc.,百度">百度(BIDU)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/NCTY.html"
	rel="suggest" title="NCTY,The9 Ltd.,第九城市">第九城市(NCTY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/JRJC.html"
	rel="suggest" title="JRJC,China Finance Online Co., Ltd.,金融界">金融界(JRJC)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CTRP.html"
	rel="suggest" title="CTRP,Ctrip.com International Ltd.,携程旅行网">携程网(CTRP)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/JOBS.html"
	rel="suggest" title="JOBS,51job, Inc.,前程无忧">前程无忧(JOBS)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/UTSI.html"
	rel="suggest" title="UTSI,UTStarcom Holdings Corp.,UT斯达康">UT斯达康(UTSI)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CNTF.html"
	rel="suggest" title="CNTF,China Techfaith Wireless Communication Technology Ltd.,泰克飞石无线技术有限公司">泰克飞石(CNTF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/SMI.html"
	rel="suggest" title="SMI,Semiconductor Manufacturing International Corp.,中芯国际">中芯国际(SMI)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/EDU.html"
	rel="suggest" title="EDU,New Oriental Education & Technology Group, Inc.,新东方教育科技集团">新东方(EDU)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/VISN.html"
	rel="suggest" title="VISN,Visionchina Media, Inc.,华视传媒">华视传媒(VISN)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/AMCN.html"
	rel="suggest" title="AMCN,AirMedia Group, Inc.,航美传媒">航美传媒(AMCN)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ACH.html"
	rel="suggest" title="ACH,Aluminum Corp. of China Ltd.,中国铝业股份有限公司">中国铝业(ACH)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CEO.html"
	rel="suggest" title="CEO,CNOOC Ltd.,中国海洋石油总公司">中海油(CEO)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CYD.html"
	rel="suggest" title="CYD,China Yuchai International Ltd.,中国玉柴国际有限公司">玉柴国际(CYD)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/GSH.html"
	rel="suggest" title="GSH,Guangshen Railway Co., Ltd.,广深铁路股份有限公司">广深铁路(GSH)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CHL.html"
	rel="suggest" title="CHL,China Mobile Ltd.,中国移动通信集团公司">中移动(CHL)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/LFC.html"
	rel="suggest" title="LFC,China Life Insurance Co. Ltd.,中国人寿保险股份有限公司">中国人寿(LFC)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/SORL.html"
	rel="suggest" title="SORL,SORL Auto Parts, Inc.,瑞立集团有限公司">瑞立集团(SORL)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CHA.html"
	rel="suggest" title="CHA,China Telecom Corp. Ltd.,中国电信集团公司">中国电信(CHA)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/SNP.html"
	rel="suggest" title="SNP,China Petroleum & Chemical Corp.,中国石油化工股份有限公司">中石化(SNP)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/SHI.html"
	rel="suggest" title="SHI,Sinopec Shanghai Petrochemical Co. Ltd.,上石化">上石化(SHI)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CEA.html"
	rel="suggest" title="CEA,China Eastern Airlines Corp. Ltd.,中国东方航空集团公司">东方航空(CEA)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/HNP.html"
	rel="suggest" title="HNP,Huaneng Power International, Inc.,华能国际电力股份有限公司">华能电力(HNP)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ZNH.html"
	rel="suggest" title="ZNH,China Southern Airlines Co. Ltd.,中国南方航空股份有限公司">南方航空(ZNH)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CSIQ.html"
	rel="suggest" title="CSIQ,Canadian Solar, Inc.,阿特斯">阿特斯(CSIQ)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CBAK.html"
	rel="suggest" title="CBAK,CBAK Energy Technology, Inc.,比克电池">比克电池(CBAK)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CAAS.html"
	rel="suggest" title="CAAS,China Automotive Systems, Inc.,中国汽车系统股份公司">中汽系统(CAAS)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CHNR.html"
	rel="suggest" title="CHNR,China Natural Resources, Inc.,中国天然资源有限公司">中国天然(CHNR)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/NDAC.html"
	rel="suggest" title="NDAC,New Dragon Asia Corp.,新龙亚洲集团">新龙亚洲(NDAC)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/FFHL.html"
	rel="suggest" title="FFHL,Fuwei Films (Holdings) Co., Ltd.,富维薄膜有限公司">富维薄膜(FFHL)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/NTE.html"
	rel="suggest" title="NTE,Nam Tai Electronics Inc.,南太电子有限公司">南太电子(NTE)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/DL.html"
	rel="suggest" title="DL,China Distance Education Holdings Ltd.,正保教育">正保教育(DL)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/STV.html"
	rel="suggest" title="STV,China Digital TV Holding Co., Ltd.,北京永新视博数字电视技术有限公司">永新视博(STV)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/XIN.html"
	rel="suggest" title="XIN,Xinyuan Real Estate Co. Ltd.,鑫苑置业">鑫苑置业(XIN)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ATV.html"
	rel="suggest" title="ATV,Acorn International, Inc.,橡果国际">橡果国际(ATV)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/SOL.html"
	rel="suggest" title="SOL,ReneSola Ltd.,浙江昱辉阳光能源有限公司">昱辉阳光(SOL)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/YGE.html"
	rel="suggest" title="YGE,Yingli Green Energy Holding Co. Ltd.,英利">英利(YGE)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CISG.html"
	rel="suggest" title="CISG,CNinsure Inc.,泛华保险">泛华保险(CISG)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/HIMX.html"
	rel="suggest" title="HIMX,Himax Technologies, Inc.,奇景光电">奇景光电(HIMX)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/JASO.html"
	rel="suggest" title="JASO,JA Solar Holdings Co., Ltd.,晶澳太阳能">晶澳太阳(JASO)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/OIIM.html"
	rel="suggest" title="OIIM,O2Micro International Ltd.,凹凸科技">凹凸科技(OIIM)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/SIMO.html"
	rel="suggest" title="SIMO,Silicon Motion Technology Corp.,慧荣科技股份有限公司">慧荣科技(SIMO)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/PTR.html"
	rel="suggest" title="PTR,PetroChina Co., Ltd.,中国石油天然气股份有限公司">中石油(PTR)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CHU.html"
	rel="suggest" title="CHU,China Unicom (Hong Kong) Ltd.,中国联通(香港)有限公司">中国联通(CHU)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ASX.html"
	rel="suggest" title="ASX,ASE Technology Holding Co., Ltd.,日月光半导体制造股份有限公司">日月光半(ASX)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/AUO.html"
	rel="suggest" title="AUO,AU Optronics Corp.,友达光电">友达光电(AUO)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/TSM.html"
	rel="suggest" title="TSM,Taiwan Semiconductor Manufacturing Co., Ltd.,台湾积体电路制造公司">台积电(TSM)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/UMC.html"
	rel="suggest" title="UMC,United Microelectronics Corp.,联华电子公司">联电(UMC)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/SPIL.html"
	rel="suggest" title="SPIL,Siliconware Precision Industries Co., Ltd.,矽品精密工业股份有限公司">矽品(SPIL)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/GIGM.html"
	rel="suggest" title="GIGM,GigaMedia Ltd.,和信超媒体公司">和信超媒(GIGM)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CYOU.html"
	rel="suggest" title="CYOU,Changyou.com Ltd.,畅游">畅游(CYOU)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/HTHT.html"
	rel="suggest" title="HTHT,Huazhu Group Ltd.,华住酒店集团">华住酒店(HTHT)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CHT.html"
	rel="suggest" title="CHT,Chunghwa Telecom Co. Ltd.,中华电信">中华电信(CHT)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/SYUT.html"
	rel="suggest" title="SYUT,Synutra International, Inc.,圣元国际">圣元国际(SYUT)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CCGM.html"
	rel="suggest" title="CCGM,China CGame, Inc.,联游网络">联游网络(CCGM)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/SFUN.html"
	rel="suggest" title="SFUN,Fang Holdings Ltd.,搜房网">搜房网(SFUN)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/DYNP.html"
	rel="suggest" title="DYNP,Duoyuan Printing Inc,多元印刷">多元印刷(DYNP)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CCIH.html"
	rel="suggest" title="CCIH,ChinaCache International Holdings Ltd.,蓝汛">蓝汛(CCIH)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/NOAH.html"
	rel="suggest" title="NOAH,Noah Holdings Ltd.,诺亚（中国）财富管理中心">诺亚财富(NOAH)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/BITA.html"
	rel="suggest" title="BITA,Bitauto Holdings Ltd.,易车网">易车网(BITA)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/XNY.html"
	rel="suggest" title="XNY,Dunxin Financial Holdings Ltd.,中国希尼亚时装有限公司">希尼亚(XNY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/KNDI.html"
	rel="suggest" title="KNDI,Kandi Technologies Group, Inc.,浙江康迪车业有限公司">康迪车业(KNDI)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CNET.html"
	rel="suggest" title="CNET,ChinaNet Online Holdings, Inc.,中网在线">中网在线(CNET)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ATAI.html"
	rel="suggest" title="ATAI,ATA, Inc.,ATA公司">ATA公司(ATAI)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/LIWA.html"
	rel="suggest" title="LIWA,Lihua International, Inc.,利华国际">利华国际(LIWA)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/YUII.html"
	rel="suggest" title="YUII,Yuhe International, Inc.,山东昱合">山东昱合(YUII)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/HEAT.html"
	rel="suggest" title="HEAT,SmartHeat, Inc.,太宇机电">太宇机电(HEAT)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/TXIC.html"
	rel="suggest" title="TXIC,Tongxin International Ltd.,同心国际">同心国际(TXIC)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/JKS.html"
	rel="suggest" title="JKS,JinkoSolar Holding Co., Ltd.,晶科能源">晶科能源(JKS)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/RINO.html"
	rel="suggest" title="RINO,RINO International Corp.,绿诺科技">绿诺科技(RINO)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/DQ.html"
	rel="suggest" title="DQ,Daqo New Energy Corp.,大全新能源">大全新能(DQ)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CRTP.html"
	rel="suggest" title="CRTP,China Ritar Power Corp,瑞达电源">瑞达电源(CRTP)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CHBT.html"
	rel="suggest" title="CHBT,China Biotics Inc,中国生物">中国生物(CHBT)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/HGSH.html"
	rel="suggest" title="HGSH,China HGS Real Estate, Inc.,汉广厦房地产">汉广厦房(HGSH)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/SEED.html"
	rel="suggest" title="SEED,Origin Agritech Ltd.,奥瑞金种业">奥瑞金种(SEED)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/BEST.html"
	rel="suggest" title="BEST,Shiner International, Inc.,赛诺国际">赛诺国际(BEST)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/WATG.html"
	rel="suggest" title="WATG,Wonder Auto Technology Inc,万得汽车">万得汽车(WATG)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CCME.html"
	rel="suggest" title="CCME,China MediaExpress Ho,中国高速传媒控股公司">中国高速(CCME)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/FUQI.html"
	rel="suggest" title="FUQI,Fuqi International Inc,福麒国际">福麒国际(FUQI)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/GURE.html"
	rel="suggest" title="GURE,Gulf Resources, Inc.,海湾资源">海湾资源(GURE)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CAST.html"
	rel="suggest" title="CAST,ChinaCast Education Corp,双威公司">双威教育(CAST)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CADC.html"
	rel="suggest" title="CADC,China Advanced Construction Materials Group, Inc.,新奥混凝土">新奥混凝(CADC)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/JFC.html"
	rel="suggest" title="JFC,JF China Region Fund Inc,JF中国基金">JF中国基(JFC)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/WUHN.html"
	rel="suggest" title="WUHN,Wuhan General Group (China), Inc.,武汉通用集团">武汉通用(WUHN)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/FEED.html"
	rel="suggest" title="FEED,AgFeed Industries Inc,艾格菲国际集团">艾格菲国(FEED)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CO.html"
	rel="suggest" title="CO,Global Cord Blood Corp.,中国脐带血库企业集团">中国脐带(CO)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/RCON.html"
	rel="suggest" title="RCON,Recon Technology Ltd.,研控科技（集团）有限公司">研控科技(RCON)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ABAT.html"
	rel="suggest" title="ABAT,Advanced Battery Technologies Inc,中强能源科技有限公司">中强能源(ABAT)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/HOLI.html"
	rel="suggest" title="HOLI,HollySys Automation Technologies Ltd.,和利时公司">和利时自(HOLI)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/KONE.html"
	rel="suggest" title="KONE,Kingtone Wirelessinfo Solution Holding Ltd.,西安联合信息技术股份有限公司">联合信息(KONE)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CIIC.html"
	rel="suggest" title="CIIC,China Infrastructure Investment Corp.,中国基础设施投资">中国基础(CIIC)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CPHI.html"
	rel="suggest" title="CPHI,China Pharma Holdings, Inc.,中国医药控股有限公司">中国医药(CPHI)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/APWR.html"
	rel="suggest" title="APWR,A Power Energy Generation Syst,高科能源">高科能源(APWR)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CREG.html"
	rel="suggest" title="CREG,China Recycling Energy Corp.,盈丰科技">盈丰科技(CREG)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/SINO.html"
	rel="suggest" title="SINO,Sino-Global Shipping America Ltd.,中环球船务">中环球船(SINO)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/JGBO.html"
	rel="suggest" title="JGBO,Jiangbo Pharmaceuticals, Inc,江波制药">江波制药(JGBO)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CHNG.html"
	rel="suggest" title="CHNG,China Natural Gas Inc,西安市西蓝天然气（集团）股份有限">西蓝公司(CHNG)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/SCEI.html"
	rel="suggest" title="SCEI,Sino Clean Energy Inc,中国清洁能源股份有限公司">中国清洁(SCEI)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CHIO.html"
	rel="suggest" title="CHIO,China INSOnline Corp,北京智远天下科技有限公司">智远天下(CHIO)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CXDC.html"
	rel="suggest" title="CXDC,China XD Plastics Co., Ltd.,鑫达集团">鑫达集团(CXDC)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CELM.html"
	rel="suggest" title="CELM,China Electric Motor. Inc,深圳岳鹏成电机有限公司">岳鹏成电(CELM)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CJJD.html"
	rel="suggest" title="CJJD,China Jo-Jo Drugstores, Inc.,中国九洲大药房">九洲大药(CJJD)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CCM.html"
	rel="suggest" title="CCM,Concord Medical Services Holding Ltd.,泰和诚医疗">泰和诚医(CCM)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/SBAY.html"
	rel="suggest" title="SBAY,Subaye, Inc.,数百亿">数百亿(SBAY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ALN.html"
	rel="suggest" title="ALN,American Lorain Corp.,绿润集团">绿润集团(ALN)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/BSPM.html"
	rel="suggest" title="BSPM,BioStar Pharmaceuticals, Inc.,奥星制药">奥星制药(BSPM)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CDII.html"
	rel="suggest" title="CDII,CD International Enterprises, Inc.,华达产业">华达产业(CDII)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CALI.html"
	rel="suggest" title="CALI,China Auto Logistics, Inc.,中国汽车物流">中国汽车(CALI)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/SVA.html"
	rel="suggest" title="SVA,Sinovac Biotech Ltd.,北京科兴生物制品有限公司">科兴生物(SVA)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CGA.html"
	rel="suggest" title="CGA,China Green Agriculture, Inc.,中国绿色农业">中国绿色(CGA)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/VALV.html"
	rel="suggest" title="VALV,Shengkai Innovations, Inc.,天津市圣恺工业技术发展有限公司">圣恺工业(VALV)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/NOEC.html"
	rel="suggest" title="NOEC,New Oriental Energy & Chemical Corp.,新东方能源化工公司">新东方能(NOEC)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/BORN.html"
	rel="suggest" title="BORN,China New Borun Corp.,中国新博润集团">博润(BORN)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ONP.html"
	rel="suggest" title="ONP,Orient Paper, Inc.,东方纸业公司">东方纸业(ONP)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/OSN.html"
	rel="suggest" title="OSN,Ossen Innovation Co., Ltd.,奥盛创新">奥盛创新(OSN)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/VNET.html"
	rel="suggest" title="VNET,21Vianet Group, Inc.,世纪互联数据中心有限公司">世纪互联(VNET)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/RENN.html"
	rel="suggest" title="RENN,Renren Inc.,人人公司">人人公司(RENN)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/FENG.html"
	rel="suggest" title="FENG,Phoenix New Media Ltd.,凤凰新媒体">凤凰新媒(FENG)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ZX.html"
	rel="suggest" title="ZX,China Zenix Auto International Ltd.,正兴集团">正兴集团(ZX)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CNIT.html"
	rel="suggest" title="CNIT,China Information Technology, Inc.,中国信息技术有限公司">中国信息(CNIT)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/VIPS.html"
	rel="suggest" title="VIPS,Vipshop Holdings Ltd.,广州唯品会信息科技有限公司">唯品会(VIPS)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/YY.html"
	rel="suggest" title="YY,YY, Inc.,欢聚时代">欢聚时代(YY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/DSWL.html"
	rel="suggest" title="DSWL,Deswell Industries, Inc.,香港德卫集团">德卫(DSWL)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/AAAE.html"
	rel="suggest" title="AAAE,AAA ENERGY INC.,AAA能源公司">AAA能源(AAAE)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/APWC.html"
	rel="suggest" title="APWC,Asia Pacific Wire & Cable Corp.,亚太电线电缆股份有限公司">亚太电线(APWC)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/IMOS.html"
	rel="suggest" title="IMOS,ChipMOS Technologies, Inc.,百慕达南茂科技股份有限公司">南茂科技(IMOS)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/AACAY.html"
	rel="suggest" title="AACAY,AAC Technologies Holdings, Inc.,瑞声科技控股有限公司">瑞声(AACAY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ACGBY.html"
	rel="suggest" title="ACGBY,Agricultural Bank of China,中国农业银行股份有限公司">中国农行(ACGBY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/AGVO.html"
	rel="suggest" title="AGVO,BUDDHA STEEL INC,宝生集团">宝生集团(AGVO)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/AHCHY.html"
	rel="suggest" title="AHCHY,Anhui Conch Cement Co., Ltd.,安徽海螺水泥股份有限公司">安徽海螺(AHCHY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/AICAF.html"
	rel="suggest" title="AICAF,Air China Ltd.,中国国际航空股份有限公司">中国国航(AICAF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/AIDA.html"
	rel="suggest" title="AIDA,AiDa Pharmaceuticals, Inc.,爱大制药有限公司">爱大制药(AIDA)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/AIVI.html"
	rel="suggest" title="AIVI,AIVtech International Group Co.,泛蓝国际集团">泛蓝国际(AIVI)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/AKRK.html"
	rel="suggest" title="AKRK,Asia Cork, Inc.,亚洲软木公司">亚洲软木(AKRK)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ALMMF.html"
	rel="suggest" title="ALMMF,Aluminum Corp. of China Ltd.,中国铝业公司">中铝公司(ALMMF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/AMVGF.html"
	rel="suggest" title="AMVGF,AMVIG Holdings Ltd.,澳科控股有限公司">澳科控股(AMVGF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ANGGY.html"
	rel="suggest" title="ANGGY,Angang Steel Co., Ltd.,鞍钢股份有限公司">鞍钢(ANGGY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CMFO.html"
	rel="suggest" title="CMFO,China Marine Food Group Ltd.,华宝明祥食品有限公司">华宝明祥(CMFO)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/AMCG.html"
	rel="suggest" title="AMCG,Amico Games Corp.,爱美柯游戏公司">爱美柯(AMCG)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/AMGY.html"
	rel="suggest" title="AMGY,American Metal & Technology, Inc.,美国金属科技公司">美国金属(AMGY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/HPJ.html"
	rel="suggest" title="HPJ,Highpower International, Inc.,豪鹏国际集团">豪鹏国际(HPJ)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/LEDS.html"
	rel="suggest" title="LEDS,SemiLEDs Corp.,旭明光电股份有限公司">旭明光电(LEDS)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ARTB.html"
	rel="suggest" title="ARTB,Art Boutique, Inc.,艺术精品公司">艺术精品(ARTB)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/BACHY.html"
	rel="suggest" title="BACHY,Bank of China Ltd.,中国银行股份有限公司">中国银行(BACHY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/BCAUY.html"
	rel="suggest" title="BCAUY,Brilliance China Automotive Holdings Ltd.,华晨中国汽车控股有限公司">华晨中国(BCAUY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/BCHM.html"
	rel="suggest" title="BCHM,Beijing Century Health Medical, Inc.,北京世纪健康医疗公司">北京世纪(BCHM)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/BCMXY.html"
	rel="suggest" title="BCMXY,Bank of Communications Co., Ltd.,交通银行">交通银行(BCMXY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/BDEV.html"
	rel="suggest" title="BDEV,Business Development Solutions Inc.,中国企业发展解决方案公司">中国企发(BDEV)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/BNSO.html"
	rel="suggest" title="BNSO,Bonso Electronics International, Inc.,恒异电子国际公司">恒异电子(BNSO)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/EVK.html"
	rel="suggest" title="EVK,Ever-Glory International Group, Inc.,江苏华瑞服装有限公司">华瑞服装(EVK)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/NEWN.html"
	rel="suggest" title="NEWN,New Energy Systems Group,新能源系统集团">新能源系(NEWN)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CNR.html"
	rel="suggest" title="CNR,China Metro-Rural Holdings Limited,中国新城农村控股有限公司">中国新城(CNR)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/BEER.html"
	rel="suggest" title="BEER,Tsingyuan Brewery Ltd.,清源啤酒有限公司">清源啤酒(BEER)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/BEIJF.html"
	rel="suggest" title="BEIJF,Beijing North Star Co. Ltd.,北京北辰实业股份有限公司">北辰实业(BEIJF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/BELLY.html"
	rel="suggest" title="BELLY,Belle International Holdings Limited,百丽国际">百丽(BELLY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/BFAR.html"
	rel="suggest" title="BFAR,BioPharm Asia Inc.,亚洲生物制药">亚洲生物(BFAR)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/BHKLY.html"
	rel="suggest" title="BHKLY,BOC Hong Kong (Holdings) Ltd.,中国银行香港控股有限公司">中银香港(BHKLY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/BJCHF.html"
	rel="suggest" title="BJCHF,Beijing Capital International Airport Co., Ltd.,北京首都国际机场">首都国际(BJCHF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/BJINY.html"
	rel="suggest" title="BJINY,Beijing Enterprises Holdings Ltd.,北京控股有限公司">北京控股(BJINY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/BKEAY.html"
	rel="suggest" title="BKEAY,The Bank of East Asia Ltd.,东亚银行有限公司">东亚银行(BKEAY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/SGOC.html"
	rel="suggest" title="SGOC,SGOCO Group Ltd.,上为集团">上为(SGOC)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CLWT.html"
	rel="suggest" title="CLWT,Euro Tech Holdings Co. Ltd.,欧陆科仪控股有限公司">欧陆科仪(CLWT)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/BOTRF.html"
	rel="suggest" title="BOTRF,China Everbright Water Ltd.,汉科环境科技集团">汉科环境(BOTRF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/BOTX.html"
	rel="suggest" title="BOTX,Bontex, Inc.,邦泰亚洲控股有限公司">邦泰(BOTX)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/BSDGY.html"
	rel="suggest" title="BSDGY,Bosideng International Holdings Ltd.,波司登国际控股有限公司">波司登(BSDGY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/BSHG.html"
	rel="suggest" title="BSHG,Bros Holding Company,百隆东方股份有限公司">百隆东方(BSHG)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/BYDDF.html"
	rel="suggest" title="BYDDF,BYD Co. Ltd.,比亚迪股份有限公司">比亚迪(BYDDF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CABLF.html"
	rel="suggest" title="CABLF,China Cablecom Holdings, Ltd.,中国 Cablecom 控股有限公司">中国 Cab(CABLF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CAGM.html"
	rel="suggest" title="CAGM,China Green Material Technologies, Inc.,中国绿色材料科技有限公司">绿色材料(CAGM)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/HIHO.html"
	rel="suggest" title="HIHO,Highway Holdings Ltd.,公路控股">公路控股(HIHO)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/KGJI.html"
	rel="suggest" title="KGJI,Kingold Jewelry, Inc.,金凰珠宝股份有限公司">金凰珠宝(KGJI)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/NFEC.html"
	rel="suggest" title="NFEC,NF Energy Saving Corp.,伟业能源科技有限公司">能发伟业(NFEC)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CAOVY.html"
	rel="suggest" title="CAOVY,China Overseas Land & Investment Ltd.,中国海外发展有限公司">中国海外(CAOVY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CARCY.html"
	rel="suggest" title="CARCY,China Resources Cement Holdings Ltd.,华润水泥控股有限公司">华润水泥(CARCY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CASDY.html"
	rel="suggest" title="CASDY,China National Materials Co. Ltd.,中国中材股份有限公司">中国中材(CASDY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CBEH.html"
	rel="suggest" title="CBEH,China Integrated Energy, Inc.,中国综合能源股份有限公司">中国综合(CBEH)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CBGH.html"
	rel="suggest" title="CBGH,China Yibai United Guarantee International Holding, Inc.,中国溢佰联合担保国际控股公司">中国溢佰(CBGH)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CBLUY.html"
	rel="suggest" title="CBLUY,China BlueChemical Ltd.,中海石油化学股份有限公司">中海石油(CBLUY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CBUMY.html"
	rel="suggest" title="CBUMY,China National Building Material Co., Ltd.,中国建材股份有限公司">中国建材(CBUMY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CBRAF.html"
	rel="suggest" title="CBRAF,CBR Brewing Company Inc.,CBR酿造股份有限公司">CBR酿造(CBRAF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/SSW.html"
	rel="suggest" title="SSW,Seaspan Corp.,西斯班公司">西斯班(SSW)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CCCGY.html"
	rel="suggest" title="CCCGY,China Communications Construction Co. Ltd.,中国交通建设股份有限公司">中交股份(CCCGY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/JMHLY.html"
	rel="suggest" title="JMHLY,Jardine Matheson Holdings Ltd.,怡和控股有限公司">怡和控股(JMHLY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CBPO.html"
	rel="suggest" title="CBPO,China Biologic Products Holdings, Inc.,泰邦生物">泰邦生物(CBPO)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/XHFNF.html"
	rel="suggest" title="XHFNF,Beat Holdings Ltd.,新华控股有限公司">新华财经(XHFNF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/HGHN.html"
	rel="suggest" title="HGHN,TEC Technology, Inc.,泰科铁塔公司">泰科铁塔(HGHN)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/JPAK.html"
	rel="suggest" title="JPAK,Jpak Group, Inc.,捷帕克集团（青岛人民印刷有限公司">青岛人印(JPAK)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/HGKGY.html"
	rel="suggest" title="HGKGY,Power Assets Holdings Ltd.,电能实业有限公司">电能实业(HGKGY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/KEYP.html"
	rel="suggest" title="KEYP,Keyuan Petrochemicals, Inc.,科元石化有限公司">科元石化(KEYP)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/HHGM.html"
	rel="suggest" title="HHGM,Huiheng Medical Incorporated Company,惠恒医疗公司">惠恒医疗(HHGM)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/KMSWF.html"
	rel="suggest" title="KMSWF,Kingmaker Footwear Holdings Ltd.,信星鞋业集团有限公司">信星鞋业(KMSWF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/HIGR.html"
	rel="suggest" title="HIGR,Hi-Great Group Holding Co.,百丰电子(上海)有限公司">百丰电子(HIGR)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/XNFHF.html"
	rel="suggest" title="XNFHF,Xi'an 38 Fule Health & Advisory Co Ltd,陕西三八妇乐科技">三八妇乐(XNFHF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/HIIDY.html"
	rel="suggest" title="HIIDY,Hidili Industry International Development Ltd.,恒鼎实业国际发展有限公司">恒鼎实业(HIIDY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/XNGSY.html"
	rel="suggest" title="XNGSY,ENN Energy Holdings Ltd.,新奥能源控股有限公司">新奥能源(XNGSY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/XSELY.html"
	rel="suggest" title="XSELY,XINHUA SPORTS & ENTMT LTD,新华悦动传媒">新华悦动(XSELY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/XYIGF.html"
	rel="suggest" title="XYIGF,Xinyi Glass Holdings Ltd.,信义玻璃控股有限公司">信义玻璃(XYIGF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/HKAEY.html"
	rel="suggest" title="HKAEY,Hong Kong Aircraft Engineering Company Limited,香港飞机工程有限公司">港机工程(HKAEY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/YINGY.html"
	rel="suggest" title="YINGY,YINGDE GASES GRP UNSP/ADR,盈德气体集团有限公司">盈德气体(YINGY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/YIPCF.html"
	rel="suggest" title="YIPCF,Yip's Chemical Holdings Ltd.,叶氏化工集团有限公司">叶氏化工(YIPCF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/JADA.html"
	rel="suggest" title="JADA,Jade Art Group, Inc.,内蒙古 佘太翠玉实业有限公司">佘太翠玉(JADA)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/YYINE.html"
	rel="suggest" title="YYINE,YAYI INTERNATIONAL INC,亚亿国际公司">亚亿国际(YYINE)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/KRYPY.html"
	rel="suggest" title="KRYPY,Kerry Properties Ltd.,嘉里建设有限公司">嘉里建设(KRYPY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ZHAOF.html"
	rel="suggest" title="ZHAOF,Zhaojin Mining Industry Co., Ltd.,招金矿业股份有限公司">招金矿业(ZHAOF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/PPCCY.html"
	rel="suggest" title="PPCCY,PICC Property & Casualty Co. Ltd.,中国人民财产保险股份有限公司">中国人财(PPCCY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/JELCY.html"
	rel="suggest" title="JELCY,Johnson Electric Holdings Ltd.,德昌电机控股有限公司">德昌电机(JELCY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/HKXCF.html"
	rel="suggest" title="HKXCF,Hong Kong Exchanges & Clearing Ltd.,香港交易所">港交所(HKXCF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/PSGP.html"
	rel="suggest" title="PSGP,Pasco Group Holding Co.,美商百胜客金融集团">美商百胜(PSGP)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/QING.html"
	rel="suggest" title="QING,Qingdao Footwear, Inc.,青岛红冠鞋业股份有限公司">红冠鞋业(QING)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/KURU.html"
	rel="suggest" title="KURU,Kun Run Biotechnology, Inc.,昆仑生物技术有限公司（音译）">昆仑生物(KURU)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/RDBO.html"
	rel="suggest" title="RDBO,Rodobo International Inc,乳多宝国际公司">乳多宝(RDBO)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/RNFU.html"
	rel="suggest" title="RNFU,Rongfu Aquaculture, Inc.,荣福水产养殖公司">荣福水产(RNFU)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/RNHEF.html"
	rel="suggest" title="RNHEF,Renhe Commercial Holdings Co. Ltd.,人和商业控股有限公司">人和商业(RNHEF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/JEXYF.html"
	rel="suggest" title="JEXYF,Jiangsu Expressway Co. Ltd.,江苏宁沪高速公路股份有限公司">江苏宁沪(JEXYF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/SEXHF.html"
	rel="suggest" title="SEXHF,Sichuan Expressway Co. Ltd.,四川成渝高速公路股份有限公司">成渝高速(SEXHF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/KWHAY.html"
	rel="suggest" title="KWHAY,K.Wah International Holdings Ltd.,嘉华国际集团有限公司">嘉华国际(KWHAY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/JEXYY.html"
	rel="suggest" title="JEXYY,Jiangsu Expressway Co. Ltd.,江苏宁沪高速公路股份有限公司">江苏宁沪(JEXYY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/SGHIY.html"
	rel="suggest" title="SGHIY,Shanghai Industrial Holdings Ltd.,上海实业控股有限公司">上海实业(SGHIY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/JIXAY.html"
	rel="suggest" title="JIXAY,Jiangxi Copper Co. Ltd.,江铜集团">江铜集团(JIXAY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/SHALY.html"
	rel="suggest" title="SHALY,Shangri-La Asia Ltd.,香格里拉（亚洲）有限公司">香格里拉(SHALY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/LEGE.html"
	rel="suggest" title="LEGE,Legend Media, Inc.,乐君传媒股份有限公司">乐君传媒(LEGE)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/SHANF.html"
	rel="suggest" title="SHANF,Shandong Molong Petroleum Machinery Co., Ltd.,山东墨龙石油机械有限公司">墨龙石油(SHANF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/SHGXY.html"
	rel="suggest" title="SHGXY,Shenguan Holdings (Group) Ltd.,神冠控股（集团）有限公司">神冠控股(SHGXY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/SHKLY.html"
	rel="suggest" title="SHKLY,Sinotruk Hong Kong Ltd.,中国重汽(香港)有限公司">重汽(香(SHKLY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/LKHLY.html"
	rel="suggest" title="LKHLY,Lonking Holdings Ltd.,龙工控股有限公司">龙工控股(LKHLY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/SHTDF.html"
	rel="suggest" title="SHTDF,Sinopharm Group Co., Ltd.,国药控股股份有限公司">国药(SHTDF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/SHTGY.html"
	rel="suggest" title="SHTGY,Shun Tak Holdings Ltd.,信得集团有限公司">信得(SHTGY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/TRUHY.html"
	rel="suggest" title="TRUHY,Truly International Holdings Ltd.,信利国际控股有限公司">信利国际(TRUHY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/TSGTF.html"
	rel="suggest" title="TSGTF,Tsingtao Brewery Co., Ltd.,青岛啤酒股份有限公司">青岛啤酒(TSGTF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/TSYHY.html"
	rel="suggest" title="TSYHY,TravelSky Technology Ltd.,中国民航信息集团公司">中国航信(TSYHY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/TTNDY.html"
	rel="suggest" title="TTNDY,Techtronic Industries Co., Ltd.,创科实业有限公司">创科实业(TTNDY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/XNWU.html"
	rel="suggest" title="XNWU,Xuan Wu International Group Holding Co.,河北玄武建材集团有限公司">玄武集团(XNWU)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/LMPMY.html"
	rel="suggest" title="LMPMY,Lee & Man Paper Manufacturing Ltd.,理文造纸有限公司">理文造纸(LMPMY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/LTUS.html"
	rel="suggest" title="LTUS,Lotus Pharmaceuticals, Inc.,莲花制药公司">莲花制药(LTUS)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/MAANF.html"
	rel="suggest" title="MAANF,Maanshan Iron & Steel Co., Ltd.,马鞍山钢铁有限公司">马鞍山钢(MAANF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/MAORF.html"
	rel="suggest" title="MAORF,Mandarin Oriental International Ltd.,文华东方国际有限公司">文化东方(MAORF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/MLLUY.html"
	rel="suggest" title="MLLUY,Metallurgical Corp. of China Ltd.,中国冶金科工股份有限公司">冶金科工(MLLUY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/NANI.html"
	rel="suggest" title="NANI,Neologic Animation Inc.,新逻辑动画公司">新逻辑动(NANI)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/TVBCY.html"
	rel="suggest" title="TVBCY,Television Broadcasts Ltd.,电视广播有限公司">无线电视(TVBCY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CEVIY.html"
	rel="suggest" title="CEVIY,China Everbright Ltd.,中国光大控股有限公司">光大控股(CEVIY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/TXNM.html"
	rel="suggest" title="TXNM,Tianxin Mining (USA) Inc,天津天鑫矿业有限公司">天鑫矿业(TXNM)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/NIVS.html"
	rel="suggest" title="NIVS,NIVS IntelliMedia Technology Group, Inc.,纳伟仕视听科技有限公司">纳伟仕视(NIVS)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/UOLI.html"
	rel="suggest" title="UOLI,Uonlive Corp.,优按网络电台">优按网络(UOLI)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CHME.html"
	rel="suggest" title="CHME,China Medicine Corp.,康采恩集团有限公司">康采恩(CHME)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/NOBGF.html"
	rel="suggest" title="NOBGF,Noble Group Ltd.,来宝集团有限公司">来宝集团(NOBGF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CCCL.html"
	rel="suggest" title="CCCL,China Ceramics Co. Ltd.,中国陶瓷">中国陶瓷(CCCL)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/WGHGY.html"
	rel="suggest" title="WGHGY,Wing Hang Bank Ltd,永亨银行有限公司">永亨银行(WGHGY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/VTSYF.html"
	rel="suggest" title="VTSYF,Vitasoy International Holdings Ltd.,维他奶集团">维他奶(VTSYF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/NWRLY.html"
	rel="suggest" title="NWRLY,New World Department Store China Ltd.,新世界百货中国有限公司">新世界百(NWRLY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/SLUXY.html"
	rel="suggest" title="SLUXY,Shandong Luoxin Pharmaceutical Group Stock Co., Ltd.,山东罗欣药业股份有限公司">罗欣药业(SLUXY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/SLVA.html"
	rel="suggest" title="SLVA,Silvan Industries Inc,贵州银燕木业有限责任公司">银燕木业(SLVA)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/SNAS.html"
	rel="suggest" title="SNAS,Sino Assurance, Inc.,中建担保有限公司">中建担保(SNAS)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/SNFRY.html"
	rel="suggest" title="SNFRY,Sinofert Holdings Ltd.,中化化肥控股有限公司">中化化肥(SNFRY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/SNLAY.html"
	rel="suggest" title="SNLAY,Sino Land Co. Ltd.,信和置业有限公司">信和置业(SNLAY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/OPEI.html"
	rel="suggest" title="OPEI,Orient Petroleum and Energy, Inc.,东方石油能源公司">东方石油(OPEI)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/UIBGF.html"
	rel="suggest" title="UIBGF,UIB Group Ltd.,北京联合保险经纪有限公司">联合经纪(UIBGF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/WGLD.html"
	rel="suggest" title="WGLD,Whole Gold International Group Holding Co,三金国际股份有限公司">三金国际(WGLD)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/HKXCY.html"
	rel="suggest" title="HKXCY,Hong Kong Exchanges & Clearing Ltd.,香港交易所">港交所(HKXCY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/HLDCY.html"
	rel="suggest" title="HLDCY,Henderson Land Development Co. Ltd.,恒基兆业地产有限公司">恒基兆业(HLDCY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/HLDVF.html"
	rel="suggest" title="HLDVF,Henderson Land Development Co. Ltd.,恒基兆业地产有限公司">恒基兆业(HLDVF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/HLPPY.html"
	rel="suggest" title="HLPPY,Hang Lung Properties Ltd.,恒隆地产有限公司">恒隆地产(HLPPY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/HMCTF.html"
	rel="suggest" title="HMCTF,Regal International Airport Group Co., Ltd.,海南美兰国际机场股份有限公司">美兰股份(HMCTF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/HNLGF.html"
	rel="suggest" title="HNLGF,Hang Lung Group Ltd.,恒隆集团有限公司">恒隆集团(HNLGF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/HNLGY.html"
	rel="suggest" title="HNLGY,Hang Lung Group Ltd.,恒隆集团有限公司">恒隆集团(HNLGY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/HOKCY.html"
	rel="suggest" title="HOKCY,Hong Kong & China Gas Co. Ltd.,香港中华煤气有限公司">煤气公司(HOKCY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CHCC.html"
	rel="suggest" title="CHCC,China Chemical Corp.,淄博嘉周化工有限公司">嘉周化工(CHCC)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/HOWWY.html"
	rel="suggest" title="HOWWY,Hopewell Holdings Ltd.,合和实业有限公司">合和实业(HOWWY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CHCG.html"
	rel="suggest" title="CHCG,China 3C Group,中国3C集团">中国3C集(CHCG)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/SRRY.html"
	rel="suggest" title="SRRY,Sancon Resources Recovery, Inc,盛棵再生资源公司">盛棵再生(SRRY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CHDA.html"
	rel="suggest" title="CHDA,China Digital Animation Development, Inc.,中国数字动画发展有限公司">中国数字(CHDA)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/HQGE.html"
	rel="suggest" title="HQGE,HQ Global Education, Inc., 环球教育股份有限公司">环球教育(HQGE)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/STTFY.html"
	rel="suggest" title="STTFY,SmarTone Telecommunications Holdings Ltd.,数码通电讯集团有限公司">数码通(STTFY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/HRELF.html"
	rel="suggest" title="HRELF,Haier Electronics Group Co., Ltd.,海尔电器集团&#8203;&#8203;">海尔(HRELF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/SUHJY.html"
	rel="suggest" title="SUHJY,Sun Hung Kai Properties Ltd.,新鸿基地产发展有限公司">新鸿基(SUHJY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CHDP.html"
	rel="suggest" title="CHDP,China Daqing M&H Petroleum, Inc.,大庆M&H石油公司">M&H石油(CHDP)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CHEUY.html"
	rel="suggest" title="CHEUY,Cheung Kong (Holdings) Limited,长江实业(集团)有限公司">长实(CHEUY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/SWRAY.html"
	rel="suggest" title="SWRAY,Swire Pacific Ltd.,太古股份有限公司">太古(SWRAY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/HRELY.html"
	rel="suggest" title="HRELY,Haier Electronics Group Co., Ltd.,海尔电器集团&#8203;&#8203;">海尔(HRELY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/SYUP.html"
	rel="suggest" title="SYUP,ANBC, Inc.,上海玉同药业">玉同药业(SYUP)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/HSNGY.html"
	rel="suggest" title="HSNGY,Hang Seng Bank Ltd.,恒生银行有限公司">恒生银行(HSNGY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/HSYT.html"
	rel="suggest" title="HSYT,Home system group,家庭系统集团">家庭系统(HSYT)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/SOUG.html"
	rel="suggest" title="SOUG,Sou 300 Group Holding Co.,搜三百集团">搜三百(SOUG)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/HTCMY.html"
	rel="suggest" title="HTCMY,Hitachi Construction Machinery Co., Ltd.,日立建机有限公司">日立建机(HTCMY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/HUABF.html"
	rel="suggest" title="HUABF,Huabao International Holdings Ltd.,华宝国际控股有限公司">华宝国际(HUABF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/HUIHY.html"
	rel="suggest" title="HUIHY,Huabao International Holdings Ltd.,华宝国际控股有限公司">华宝国际(HUIHY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/HUIY.html"
	rel="suggest" title="HUIY,Hui Ying Technology & Media Group Holding Co,山东惠影科技传媒股份有限公司">惠影科技(HUIY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/HYSNY.html"
	rel="suggest" title="HYSNY,Hysan Development Co., Ltd.,希慎兴业有限公司">希慎兴业(HYSNY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/IDCBF.html"
	rel="suggest" title="IDCBF,Industrial & Commercial Bank of China Ltd.,中国工商银行股份有限公司">工商银行(IDCBF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/IDCBY.html"
	rel="suggest" title="IDCBY,Industrial & Commercial Bank of China Ltd.,中国工商银行股份有限公司">工商银行(IDCBY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/IDCX.html"
	rel="suggest" title="IDCX,North China Horticulture, Inc,华北园艺公司">华北园艺(IDCX)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/IICN.html"
	rel="suggest" title="IICN,China Intelligence Information Systems, Inc.,中国情报信息系统公司">情报系统(IICN)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/IINHY.html"
	rel="suggest" title="IINHY,Imagi International Holdings Limited,意马国际控股有限公司">意马(IINHY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/IKTSY.html"
	rel="suggest" title="IKTSY,Intertek Group Plc,天祥集团">天祥集团(IKTSY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/XYIGY.html"
	rel="suggest" title="XYIGY,Xinyi Glass Holdings Ltd.,信义玻璃控股有限公司">信义玻璃(XYIGY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/TSGTY.html"
	rel="suggest" title="TSGTY,Tsingtao Brewery Co., Ltd.,青岛啤酒股份有限公司">青岛啤酒(TSGTY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CHPN.html"
	rel="suggest" title="CHPN,China Polypeptide Group Inc.,中国多肽产业集团">中国多肽(CHPN)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CHPXY.html"
	rel="suggest" title="CHPXY,China Pacific Insurance (Group) Co. Ltd.,中国太平洋保险（集团）股份有限公">中国太保(CHPXY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CHWTF.html"
	rel="suggest" title="CHWTF,Coolpad Group Ltd.,中国无线科技有限公司">中国无线(CHWTF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CHHE.html"
	rel="suggest" title="CHHE,China Health Industries Holdings, Inc.,中国健康产业集团股份有限公司">健康产业(CHHE)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CHSY.html"
	rel="suggest" title="CHSY,China Medical Systems, Inc.,中国康哲药业公司">康哲药业(CHSY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CICHY.html"
	rel="suggest" title="CICHY,China Construction Bank Corp.,中国建设银行股份有限公司">中国建设(CICHY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ZHDM.html"
	rel="suggest" title="ZHDM,Zhong Hui Dao Ming Copper Holding Ltd.,中汇道明铜业控股有限公司">中汇道明(ZHDM)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ZHEXF.html"
	rel="suggest" title="ZHEXF,Zhejiang Expressway Co. Ltd.,浙江沪杭甬高速公路股份有限公司">沪杭甬高(ZHEXF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CITAY.html"
	rel="suggest" title="CITAY,COSCO SHIPPING Development Co., Ltd.,中海集装箱运输有限公司">中海集装(CITAY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ZHYLF.html"
	rel="suggest" title="ZHYLF,Zhaoheng Hydropower Ltd.,兆恒水电有限公司">兆恒水电(ZHYLF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CICOY.html"
	rel="suggest" title="CICOY,COSCO SHIPPING Holdings Co. Ltd.,中国远洋控股股份有限公司">中国远洋(CICOY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CIDHY.html"
	rel="suggest" title="CIDHY,China Agri-Industries Holdings Ltd.,中国粮油控股有限公司">中粮(CIDHY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CHSTY.html"
	rel="suggest" title="CHSTY,China High Speed Transmission Equipment Group Co., Ltd.,中国高速传动设备集团有限公司">中国传动(CHSTY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CIVN.html"
	rel="suggest" title="CIVN,China Interactive Education Inc,中国协同教学(集团)有限公司">协同教学(CIVN)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CIFHY.html"
	rel="suggest" title="CIFHY,China Fishery Group Ltd.,中渔集团有限公司">中渔集团(CIFHY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CIHD.html"
	rel="suggest" title="CIHD,Changda International Holdings, Inc.,昌大国际控股有限公司">常达国际(CIHD)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CIHKY.html"
	rel="suggest" title="CIHKY,China Merchants Bank Co., Ltd.,招商银行">招商银行(CIHKY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ZIJMF.html"
	rel="suggest" title="ZIJMF,Zijin Mining Group Co., Ltd.,紫金矿业集团股份有限公司">紫金矿业(ZIJMF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CHOR.html"
	rel="suggest" title="CHOR,China Organic Fertilizer, Inc.,中国有机肥料">中国有机(CHOR)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ZLIOY.html"
	rel="suggest" title="ZLIOY,Zoomlion Heavy Industry Science & Technology Co. Ltd.,中联重科股份有限公司">中联重科(ZLIOY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ZTCOF.html"
	rel="suggest" title="ZTCOF,ZTE Corp.,中兴通讯股份有限公司">中兴通讯(ZTCOF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CHLBF.html"
	rel="suggest" title="CHLBF,China Health Labs & Diagnostics Ltd.,中国卫生实验室和诊断有限公司">中国卫生(CHLBF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CIWT.html"
	rel="suggest" title="CIWT,China Industrial Waste Management, Inc.,中国工业废弃物管理公司">中国工业(CIWT)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ZYCI.html"
	rel="suggest" title="ZYCI,ZIYANG CERAMICS CORP,诸城市紫阳陶瓷有限公司">紫阳陶瓷(ZYCI)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/TAAI.html"
	rel="suggest" title="TAAI,Tombao Antiques & Art Group,古今通宝艺术品国际控股集团公司">古今通宝(TAAI)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/TCEHY.html"
	rel="suggest" title="TCEHY,Tencent Holdings Ltd.,腾讯公司">腾讯(TCEHY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/TCEPY.html"
	rel="suggest" title="TCEPY,Tianjin Capital Environmental Protection Group Co., Ltd.,天津创业环保股份有限公司">创业环保(TCEPY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/TCYMF.html"
	rel="suggest" title="TCYMF,Tingyi (Cayman Islands) Holding Corp.,康师傅控股有限公司">康师傅(TCYMF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CKGT.html"
	rel="suggest" title="CKGT,China Kangtai Cactus Bio-Tech Inc.,中国康太仙人掌生物科技公司">康太仙人(CKGT)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CKUN.html"
	rel="suggest" title="CKUN,China Shenghuo Pharmaceutical Holdings, Inc.,中国圣火药业（集团）有限公司">圣火药业(CKUN)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/SHWGY.html"
	rel="suggest" title="SHWGY,Shandong Weigao Group Medical Polymer Co. Ltd.,山东威高集团医用高分子制品股份有">山东威高(SHWGY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/SHZNY.html"
	rel="suggest" title="SHZNY,Shenzhen Expressway Co., Ltd.,深圳高速公路股份有限公司">深圳高速(SHZNY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/SIMTF.html"
	rel="suggest" title="SIMTF,SIM Technology Group Ltd.,晨讯科技集团">晨讯科技(SIMTF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/OROVY.html"
	rel="suggest" title="OROVY,Orient Overseas (International) Ltd.,东方海外国际有限公司">东方海外(OROVY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/TCTZF.html"
	rel="suggest" title="TCTZF,Tencent Holdings Ltd.,腾讯公司">腾讯(TCTZF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/TCYMY.html"
	rel="suggest" title="TCYMY,Tingyi (Cayman Islands) Holding Corp.,康师傅控股有限公司">康师傅(TCYMY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/SIELY.html"
	rel="suggest" title="SIELY,Shanghai Electric Group Co., Ltd.,上海电气集团股份有限公司">上海电气(SIELY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/SIOLY.html"
	rel="suggest" title="SIOLY,Sino-Ocean Group Holdings Ltd.,远洋地产控股有限公司">远洋地产(SIOLY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/TRUHF.html"
	rel="suggest" title="TRUHF,Truly International Holdings Ltd.,信利国际有限公司">信利国际(TRUHF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/SWRBY.html"
	rel="suggest" title="SWRBY,Swire Pacific Ltd.,太古股份有限公司">太古(SWRBY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ORSX.html"
	rel="suggest" title="ORSX,Orsus Xelent Technologies, Inc.,奥盛技术股份有限公司">奥盛技术(ORSX)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/PBEP.html"
	rel="suggest" title="PBEP,Pacific Bepure Industry, Inc.,太平洋宝飘实业有限公司">太平洋宝(PBEP)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CHNC.html"
	rel="suggest" title="CHNC,China Infrastructure Construction Corp.,中国基础设施建设公司">中基建(CHNC)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/PIAIF.html"
	rel="suggest" title="PIAIF,Ping An Insurance (Group) Co. of China Ltd.,平安保险(集团)股份有限公司">平安保险(PIAIF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/PKSGY.html"
	rel="suggest" title="PKSGY,Parkson Retail Group Ltd.,百盛商业集团有限公司">百盛商业(PKSGY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/KSTV.html"
	rel="suggest" title="KSTV,KSTV Holding Co,上海坤伦文化传播有限公司">坤伦文化(KSTV)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/WOSSF.html"
	rel="suggest" title="WOSSF,Water Oasis Group Ltd.,奥思集团">奥思集团(WOSSF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/WSYS.html"
	rel="suggest" title="WSYS,Westergaard.com Inc,韦斯特加德公司">韦斯特加(WSYS)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/WTFS.html"
	rel="suggest" title="WTFS,Xinde Technology Co,潍坊信德燃油喷射系统有限公司">信德燃喷(WTFS)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/WUMSF.html"
	rel="suggest" title="WUMSF,Wumart Stores Inc,北京物美集团">北京物美(WUMSF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/WWNTF.html"
	rel="suggest" title="WWNTF,Want Want China Holdings Ltd.,旺旺中国控股有限公司">旺旺中国(WWNTF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/WYNMF.html"
	rel="suggest" title="WYNMF,Wynn Macau Ltd.,永利澳门有限公司">永利澳门(WYNMF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ZHEXY.html"
	rel="suggest" title="ZHEXY,ZHEJIANG EXPRESSWAY H,浙江沪杭甬高速公路股份有限公司">沪杭甬高(ZHEXY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ZIJMY.html"
	rel="suggest" title="ZIJMY,Zijin Mining Group Co., Ltd.,紫金矿业集团股份有限公司">紫金矿业(ZIJMY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ZTCOY.html"
	rel="suggest" title="ZTCOY,ZTE Corp.,中兴通讯股份有限公司">中兴通讯(ZTCOY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/WUMSY.html"
	rel="suggest" title="WUMSY,Wumart Stores, Inc.,北京物美集团">北京物美(WUMSY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/WWNTY.html"
	rel="suggest" title="WWNTY,Want Want China Holdings Ltd.,旺旺中国控股有限公司">旺旺中国(WWNTY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/WYNMY.html"
	rel="suggest" title="WYNMY,Wynn Macau Ltd.,永利澳门有限公司">永利澳门(WYNMY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/MNOIY.html"
	rel="suggest" title="MNOIY,Mandarin Oriental International Ltd.,文华东方国际有限公司">文化东方(MNOIY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/NOBGY.html"
	rel="suggest" title="NOBGY,Noble Group Ltd.,来宝集团有限公司">来宝集团(NOBGY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CLPXY.html"
	rel="suggest" title="CLPXY,China Longyuan Power Group Corp. Ltd.,中国龙源电力股份有限公司">龙源电力(CLPXY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CMAKY.html"
	rel="suggest" title="CMAKY,China Minsheng Banking Corp., Ltd.,中国民生银行">民生银行(CMAKY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CMNW.html"
	rel="suggest" title="CMNW,China M161 Network Co,中国M161网络公司">中国M161(CMNW)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/PNGAY.html"
	rel="suggest" title="PNGAY,Ping An Insurance (Group) Co. of China Ltd.,平安保险(集团)股份有限公司">平安保险(PNGAY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CNER.html"
	rel="suggest" title="CNER,China New Energy Group Co.,中国新能源集团">中国新能(CNER)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CPCAY.html"
	rel="suggest" title="CPCAY,Cathay Pacific Airways Ltd.,国泰航空公司">国泰航空(CPCAY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CPDV.html"
	rel="suggest" title="CPDV,China Properties Developments,中国房地产发展有限公司">中房发展(CPDV)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CPKPF.html"
	rel="suggest" title="CPKPF,CP Pokphand Co. Ltd.,卜蜂国际有限公司">卜蜂国际(CPKPF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CRHKY.html"
	rel="suggest" title="CRHKY,China Resources Beer (Holdings) Co. Ltd.,华润创业有限公司">华润(CRHKY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CRHO.html"
	rel="suggest" title="CRHO,Chenghui Realty Holding Co,广西南宁呈辉置业有限公司">呈辉置业(CRHO)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CRJI.html"
	rel="suggest" title="CRJI,China Runji Cement, Inc.,中国润基水泥有限公司">润基水泥(CRJI)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CRUI.html"
	rel="suggest" title="CRUI,China RuiTai International Holdings Co. Ltd.,中国瑞泰国际股份有限公司">瑞泰国际(CRUI)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CHOLY.html"
	rel="suggest" title="CHOLY,China Oilfield Services Ltd.,中海油田服务股份有限公司">中海油服(CHOLY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CHOLF.html"
	rel="suggest" title="CHOLF,China Oilfield Services Ltd.,中海油田服务股份有限公司">中海油服(CHOLF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CRWOY.html"
	rel="suggest" title="CRWOY,China Railway Group Ltd.,中国中铁股份有限公司">中国中铁(CRWOY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CSDXY.html"
	rel="suggest" title="CSDXY,China Shipping Development Co. Ltd.,中海发展股份有限公司">中海发展(CSDXY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CSEHY.html"
	rel="suggest" title="CSEHY,China Solar Energy Holdings Limited,中国源畅光电能源控股有限公司">源畅光电(CSEHY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CSGH.html"
	rel="suggest" title="CSGH,China Sun Group High-Tech Co.,大连新阳高科技发展有限公司">大连新阳(CSGH)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CSGJ.html"
	rel="suggest" title="CSGJ,China Shuangji Cement, Ltd.,中国双吉水泥有限公司">双吉水泥(CSGJ)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CSKD.html"
	rel="suggest" title="CSKD,China Skyrise Digital Service, Inc.,中国兴天下数字服务有限公司">兴天下(CSKD)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CCGY.html"
	rel="suggest" title="CCGY,China Clean Energy, Inc.,福建中德能源">福建中德(CCGY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CSNH.html"
	rel="suggest" title="CSNH,China Shandong Industries, Inc.,山东省曹普工艺有限公司">曹普工艺(CSNH)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CSOL.html"
	rel="suggest" title="CSOL,China Solar & Clean Energy Solutions, Inc.,德利国际新能源控股有限公司">德利国际(CSOL)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CSRGY.html"
	rel="suggest" title="CSRGY,CSR CORP LTD-UNSP ADR,中国南车股份有限公司">中国南车(CSRGY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CSUAY.html"
	rel="suggest" title="CSUAY,China Shenhua Energy Co., Ltd.,中国神华能源股份有限公司">中国神华(CSUAY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CSWG.html"
	rel="suggest" title="CSWG,Sen Yu International Holdings Inc,森宇国际控股有限公司">森宇国际(CSWG)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CSWYY.html"
	rel="suggest" title="CSWYY,China Shineway Pharmaceutical Group Ltd.,中国神威药业集团有限公司">神威药业(CSWYY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CTEK.html"
	rel="suggest" title="CTEK,CynergisTek, Inc.,辽宁新兴佳创新公司">新兴佳(CTEK)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CTHL.html"
	rel="suggest" title="CTHL,China Tractor Holdings, Inc,中国拖拉机控股公司">中拖(CTHL)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CTPCY.html"
	rel="suggest" title="CTPCY,CITIC Ltd.,中信泰富有限公司">中信泰富(CTPCY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CTVZ.html"
	rel="suggest" title="CTVZ,China Travel Resort Holdings, Inc.,中国旅游度假村控股公司">中国旅游(CTVZ)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CPKPY.html"
	rel="suggest" title="CPKPY,CP Pokphand Co. Ltd.,卜蜂国际有限公司">卜蜂国际(CPKPY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/HUAZ.html"
	rel="suggest" title="HUAZ,Hua Ye Gas Group Holding Co.,河北华野燃气集团公司">华野燃气(HUAZ)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CCOZY.html"
	rel="suggest" title="CCOZY,China Coal Energy Co., Ltd.,中国中煤能源股份有限公司">中煤能源(CCOZY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CCOZF.html"
	rel="suggest" title="CCOZF,China Coal Energy Co., Ltd.,中国中煤能源股份有限公司">中煤能源(CCOZF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CDBH.html"
	rel="suggest" title="CDBH,China Domestica Bio-technology Holdings, Inc.,中国生物技术控股公司">中国生物(CDBH)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CDBT.html"
	rel="suggest" title="CDBT,China Dasheng Biotechnology Co.,甘肃大圣生物科技股份有限公司">甘肃大圣(CDBT)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CDGXF.html"
	rel="suggest" title="CDGXF,China Dongxiang (Group) Co. Ltd.,中国动向(集团)有限公司">中国动向(CDGXF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CDSG.html"
	rel="suggest" title="CDSG,China DongSheng International, Inc.,东升伟业生物工程集团有限公司">东升伟业(CDSG)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/AIRYY.html"
	rel="suggest" title="AIRYY,Air China Ltd.,中国国际航空股份有限公司">中国国航(AIRYY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/BACHF.html"
	rel="suggest" title="BACHF,Bank of China Ltd.,中国银行股份有限公司">中国银行(BACHF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/BCAUF.html"
	rel="suggest" title="BCAUF,Brilliance China Automotive Holdings Ltd.,华晨中国汽车控股有限公司">华晨中国(BCAUF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/BKEAF.html"
	rel="suggest" title="BKEAF,The Bank of East Asia Ltd.,东亚银行有限公司">东亚银行(BKEAF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CICHF.html"
	rel="suggest" title="CICHF,China Construction Bank Corp.,中国建设银行股份有限公司">中国建设(CICHF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/BYDDY.html"
	rel="suggest" title="BYDDY,BYD Co. Ltd.,比亚迪股份有限公司">比亚迪(BYDDY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/DPNEY.html"
	rel="suggest" title="DPNEY,Daphne International Holdings Ltd.,达芙妮集团资讯及购物中心">达芙妮集(DPNEY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/DRCR.html"
	rel="suggest" title="DRCR,Dear Cashmere Holding Co.,美国�爱尔羊绒集团控股公司">�爱尔羊(DRCR)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/DUKS.html"
	rel="suggest" title="DUKS,ANHUI TAIYANG POULTRY CO,安徽太阳禽业有限公司">太阳禽业(DUKS)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/DXIN.html"
	rel="suggest" title="DXIN,Dong Xin Bio-Tech Pharmaceutical, Inc.,廊坊东信生物科技有限公司">东信生物(DXIN)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/EAST.html"
	rel="suggest" title="EAST,Eastside Distilling, Inc.,东方安全防范服务有限公司">东方安防(EAST)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/EESC.html"
	rel="suggest" title="EESC,Eastern Environment Solutions Corp,哈尔滨亿丰生态环境有限公司">亿丰环境(EESC)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/EKONY.html"
	rel="suggest" title="EKONY,E-KONG GROUP LTD S,e-KONG集团">e-KONG集(EKONY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ENHD.html"
	rel="suggest" title="ENHD,Energroup Holdings Corp.,大连础明集团有限公司">大连础明(ENHD)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ESPGY.html"
	rel="suggest" title="ESPGY,Esprit Holdings Ltd.,思捷环球控股有限公司">思捷环球(ESPGY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/FEPU.html"
	rel="suggest" title="FEPU,FLYING EAGLE PU TECH CORP,石狮飞鹰塑胶有限公司">飞鹰塑胶(FEPU)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/FIRRY.html"
	rel="suggest" title="FIRRY,First Tractor Co., Ltd.,第一拖拉机股份有限公司">第一拖拉(FIRRY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/FLMS.html"
	rel="suggest" title="FLMS,FLM MINERALS INC,FLM矿业股份有限公司">FLM矿业(FLMS)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/FPAFF.html"
	rel="suggest" title="FPAFF,First Pacific Co. Ltd.,第一太平洋控股有限公司">第一太平(FPAFF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/FXCNY.html"
	rel="suggest" title="FXCNY,FIH Mobile Ltd.,富士康国际控股有限公司">富士康控(FXCNY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/GGDVY.html"
	rel="suggest" title="GGDVY,Guangdong Investment Ltd.,粤海投资有限公司">粤海投资(GGDVY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/GHII.html"
	rel="suggest" title="GHII,Invesco S&P High Income Infrastructure ETF,金马国际公司">金马国际(GHII)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/GIGNF.html"
	rel="suggest" title="GIGNF,Genting Singapore Ltd.,云顶新加坡有限公司">云顶新加(GIGNF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/GIGNY.html"
	rel="suggest" title="GIGNY,Genting Singapore Ltd.,云顶新加坡有限公司">云顶新加(GIGNY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/GMDTF.html"
	rel="suggest" title="GMDTF,Golden Meditech Holdings Ltd.,金卫医疗集团有限公司">金卫医疗(GMDTF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/GOEG.html"
	rel="suggest" title="GOEG,Golden Elephant Glass Technology, Inc,金象玻璃科技公司">金象玻璃(GOEG)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/GPHG.html"
	rel="suggest" title="GPHG,Global Pharm Holdings Group, Inc.,环球医药控股集团">环球医药(GPHG)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/GTVI.html"
	rel="suggest" title="GTVI,Joway Health Industries Group, Inc.,中威盛世集团">中威盛世(GTVI)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/GWLLY.html"
	rel="suggest" title="GWLLY,Great Wall Motor Co., Ltd.,长城汽车股份有限公司">长城汽车(GWLLY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/GXYEF.html"
	rel="suggest" title="GXYEF,Galaxy Entertainment Group Ltd.,银河娱乐集团有限公司">银河娱乐(GXYEF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/GXYEY.html"
	rel="suggest" title="GXYEY,Galaxy Entertainment Group Ltd,银河娱乐集团有限公司">银河娱乐(GXYEY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/GZPHY.html"
	rel="suggest" title="GZPHY,Guangzhou Pharmaceutical Co Ltd,广州药业股份有限公司">广州药业(GZPHY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/GZUHY.html"
	rel="suggest" title="GZUHY,Guangzhou R&F Properties Co., Ltd.,广州富力地产股份有限公司">广州富力(GZUHY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/HDVTY.html"
	rel="suggest" title="HDVTY,Henderson Investment Ltd.,恒基兆业地产有限公司">恒基兆业(HDVTY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/HEGIF.html"
	rel="suggest" title="HEGIF,Hengan International Group Co., Ltd.,恒安国际集团有限公司">恒安国际(HEGIF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/HEGIY.html"
	rel="suggest" title="HEGIY,Hengan International Group Co., Ltd.,恒安国际集团有限公司">恒安国际(HEGIY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/FPAFY.html"
	rel="suggest" title="FPAFY,First Pacific Co. Ltd.,第一太平洋控股有限公司">第一太平(FPAFY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CVPH.html"
	rel="suggest" title="CVPH,China Vitup Health Care Holdings In,中国维特奥健康管理控股有限公司">维特奥健(CVPH)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CWPWF.html"
	rel="suggest" title="CWPWF,Concord New Energy Group Ltd.,中国风电集团有限公司">中国风电(CWPWF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CWYCY.html"
	rel="suggest" title="CWYCY,China Railway Construction Corp. Ltd.,中国铁建股份有限公司">中国铁建(CWYCY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CYDI.html"
	rel="suggest" title="CYDI,Cybrdi, Inc.,超英生物有限公司">超英生物(CYDI)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CYHM.html"
	rel="suggest" title="CYHM,CHYF Media Group Holding Co,中国海逸风传媒集团控股公司">海逸风传(CYHM)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CYIG.html"
	rel="suggest" title="CYIG,Spring Pharmaceutical Group, Inc.,中国永春堂国际集团有限公司">永春堂(CYIG)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/DCRD.html"
	rel="suggest" title="DCRD,Decor Products International Inc,东莞装饰产品国际公司">东莞饰纸(DCRD)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/DFEL.html"
	rel="suggest" title="DFEL,China TMK Battery Systems, Inc.,深圳市三俊电池有限公司">三俊电池(DFEL)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/DFIHY.html"
	rel="suggest" title="DFIHY,Dairy Farm International Holdings Ltd.,牛奶国际控股有限公司">牛奶国际(DFIHY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/DGDH.html"
	rel="suggest" title="DGDH,Dadongnan Holding Co,大东南生命水科技有限公司">大东南(DGDH)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/DGWIY.html"
	rel="suggest" title="DGWIY,Duoyuan Global Water Inc.,多元环球水务有限公司">多元环球(DGWIY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/DION.html"
	rel="suggest" title="DION,Dionics, Inc.,上饶百花洲实业有限公司">百花洲实(DION)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/DIPGY.html"
	rel="suggest" title="DIPGY,Datang International Power Generation Co., Ltd.,大唐国际发电股份有限公司">大唐发电(DIPGY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/DKSP.html"
	rel="suggest" title="DKSP,DK Sinopharma Inc,西安东科药业">东科药业(DKSP)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/DNFGF.html"
	rel="suggest" title="DNFGF,Dongfeng Motor Group Co., Ltd.,Dongfeng Motor Group Co Ltd">东风汽车(DNFGF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/DNGH.html"
	rel="suggest" title="DNGH,Dongsheng Pharmaceutical International Co. Ltd.,东盛医药国际有限公司">东盛医药(DNGH)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/DPNEF.html"
	rel="suggest" title="DPNEF,Daphne International Holdings Ltd.,达芙妮集团资讯及购物中心">达芙妮集(DPNEF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/DNFGY.html"
	rel="suggest" title="DNFGY,Dongfeng Motor Group Co., Ltd.,Dongfeng Motor Group Co Ltd">东风汽车(DNFGY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/LITB.html"
	rel="suggest" title="LITB,LightInTheBox Holding Co., Ltd.,兰亭集势控股有限责任公司">兰亭集势(LITB)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/PME.html"
	rel="suggest" title="PME,Pingtan Marine Enterprise, Ltd.,平潭海洋公司">平潭海洋(PME)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CCCR.html"
	rel="suggest" title="CCCR,China Commercial Credit, Inc.,吴江鲈乡农村小额信贷股份有限公司">鲈乡小贷(CCCR)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/WUBA.html"
	rel="suggest" title="WUBA,58.com Inc.,58同城">58同城(WUBA)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/WBAI.html"
	rel="suggest" title="WBAI,500.com Ltd.,500彩票网">500彩票(WBAI)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ATHM.html"
	rel="suggest" title="ATHM,Autohome, Inc.,汽车之家">汽车之家(ATHM)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/AMC.html"
	rel="suggest" title="AMC,AMC Entertainment Holdings, Inc.,AMC院线公司">AMC院线(AMC)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/NVFY.html"
	rel="suggest" title="NVFY,Nova Lifestyle, Inc.,诺华家具有限公司">诺华家具(NVFY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/WB.html"
	rel="suggest" title="WB,Weibo Corp.,微博">微博(WB)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/TEDU.html"
	rel="suggest" title="TEDU,Tarena International, Inc.,达内科技">达内科技(TEDU)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/LEJU.html"
	rel="suggest" title="LEJU,Leju Holdings Ltd.,乐居控股有限公司">乐居(LEJU)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/KANG.html"
	rel="suggest" title="KANG,iKang Healthcare Group, Inc.,爱康国宾">爱康国宾(KANG)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/JD.html"
	rel="suggest" title="JD,JD.com, Inc.,京东">京东(JD)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/TOUR.html"
	rel="suggest" title="TOUR,Tuniu Corp.,途牛旅游网">途牛(TOUR)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CMCM.html"
	rel="suggest" title="CMCM,Cheetah Mobile, Inc.,猎豹移动公司">猎豹移动(CMCM)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/GOAS.html"
	rel="suggest" title="GOAS,Goa Sweet Tours Ltd.,祥天控股集团">祥天控股(GOAS)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/JMEI.html"
	rel="suggest" title="JMEI,Jumei International Holding Ltd.,聚美优品">聚美优品(JMEI)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/XNET.html"
	rel="suggest" title="XNET,Xunlei Ltd.,迅雷">迅雷(XNET)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/BABA.html"
	rel="suggest" title="BABA,Alibaba Group Holding Ltd.,阿里巴巴">阿里巴巴(BABA)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/SKYS.html"
	rel="suggest" title="SKYS,Sky Solar Holdings Ltd.,天华阳光">天华阳光(SKYS)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/EHIC.html"
	rel="suggest" title="EHIC,eHi Car Services Ltd.,一嗨租车">一嗨租车(EHIC)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/MOMO.html"
	rel="suggest" title="MOMO,Momo, Inc.,陌陌科技公司">陌陌(MOMO)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/HLG.html"
	rel="suggest" title="HLG,Hailiang Education Group, Inc.,海亮教育集团">海亮教育(HLG)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/JP.html"
	rel="suggest" title="JP,Jupai Holdings Ltd.,钜派投资集团">钜派(JP)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/YRD.html"
	rel="suggest" title="YRD,Yirendai Ltd.,宜人贷公司">宜人贷(YRD)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/BGNE.html"
	rel="suggest" title="BGNE,BeiGene Ltd.,百济神州公司">百济神州(BGNE)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/YIN.html"
	rel="suggest" title="YIN,Yintech Investment Holdings Ltd.,银科投资控股有限公司">银科控股(YIN)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/COE.html"
	rel="suggest" title="COE,China Online Education Group,China Online Education Group">无忧英语(COE)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/JMU.html"
	rel="suggest" title="JMU,JMU Ltd.,众美联">众美联(JMU)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/GSUM.html"
	rel="suggest" title="GSUM,Gridsum Holding, Inc.,北京国双科技有限公司">国双科技(GSUM)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CLDC.html"
	rel="suggest" title="CLDC,China Lending Corp.,中国贷款公司">中国贷款(CLDC)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ZTO.html"
	rel="suggest" title="ZTO,ZTO Express (Cayman) Inc.,中通快递">中通(ZTO)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/XRF.html"
	rel="suggest" title="XRF,China Rapid Finance Ltd.,中国信而富公司">信而富(XRF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/BEDU.html"
	rel="suggest" title="BEDU,Bright Scholar Education Holdings Ltd.,博实乐教育集团">博实乐(BEDU)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CIFS.html"
	rel="suggest" title="CIFS,China Internet Nationwide Financial Services, Inc.,圣盈信">圣盈信(CIFS)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/BSTI.html"
	rel="suggest" title="BSTI,BEST, Inc. (China),百世集团">百世(BSTI)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/SECO.html"
	rel="suggest" title="SECO,Secoo Holding Ltd.,寺库集团">寺库(SECO)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/RYB.html"
	rel="suggest" title="RYB,RYB Education, Inc.,红黄蓝儿童教育科技发展有限公司">红黄蓝(RYB)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/REDU.html"
	rel="suggest" title="REDU,RISE Education Cayman Ltd.,瑞思学科英语">瑞思(REDU)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/SOGO.html"
	rel="suggest" title="SOGO,Sogou, Inc.,搜狗公司">搜狗(SOGO)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/PPDF.html"
	rel="suggest" title="PPDF,PPDAI Group, Inc.,拍拍贷集团公司">拍拍贷(PPDF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/QD.html"
	rel="suggest" title="QD,Qudian Inc.,趣店集团">趣店(QD)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/HX.html"
	rel="suggest" title="HX,Hexindai Inc.,和信贷公司">和信贷(HX)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/JT.html"
	rel="suggest" title="JT,Jianpu Technology, Inc.,简普科技公司">简普科技(JT)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/LX.html"
	rel="suggest" title="LX,Lexinfintech Holdings Ltd.,乐信集团">乐信(LX)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/HMI.html"
	rel="suggest" title="HMI,Huami Corp.,华米科技">华米(HMI)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/LKM.html"
	rel="suggest" title="LKM,Link Motion, Inc.,凌动智行有限公司">凌动智行(LKM)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/STG.html"
	rel="suggest" title="STG,Sunlands Online Education Group,尚德在线教育科技公司">尚德机构(STG)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/IQ.html"
	rel="suggest" title="IQ,iQIYI, Inc.,爱奇艺公司">爱奇艺(IQ)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/BILI.html"
	rel="suggest" title="BILI,Bilibili, Inc.,哔哩哔哩公司">哔哩哔哩(BILI)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/GHG.html"
	rel="suggest" title="GHG,Greentree Hospitality Group Ltd.,格林酒店集团">格林酒店(GHG)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/HUYA.html"
	rel="suggest" title="HUYA,HUYA, Inc.,虎牙公司">虎牙(HUYA)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/UXIN.html"
	rel="suggest" title="UXIN,Uxin Ltd.,优信集团">优信(UXIN)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/PDD.html"
	rel="suggest" title="PDD,Pinduoduo, Inc.,拼多多公司">拼多多(PDD)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/JG.html"
	rel="suggest" title="JG,Aurora Mobile Ltd.,极光大数据">极光(JG)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/PT.html"
	rel="suggest" title="PT,Pintec Technology Holdings Ltd.,品钛科技控股公司">品钛(PT)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/YI.html"
	rel="suggest" title="YI,111, Inc.,1药网">1药网(YI)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/NIO.html"
	rel="suggest" title="NIO,NIO, Inc. (China),蔚来汽车公司">蔚来(NIO)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/QTT.html"
	rel="suggest" title="QTT,Qutoutiao, Inc.,趣头条公司">趣头条(QTT)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/VIOT.html"
	rel="suggest" title="VIOT,Viomi Technology Co., Ltd.,云米科技有限公司">云米(VIOT)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/XYF.html"
	rel="suggest" title="XYF,X Financial,小赢科技公司">小赢科技(XYF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/LAIX.html"
	rel="suggest" title="LAIX,LAIX, Inc.,流利说信息技术有限公司">流利说(LAIX)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CTK.html"
	rel="suggest" title="CTK,CooTek (Cayman) Inc.,触宝信息技术公司">触宝(CTK)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/NIU.html"
	rel="suggest" title="NIU,Niu Technologies,小牛电动">小牛电动(NIU)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CNF.html"
	rel="suggest" title="CNF,CNFinance Holdings Ltd.,泛华金融控股公司">泛华金融(CNF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/WEI.html"
	rel="suggest" title="WEI,Weidai Ltd.,微贷网">微贷网(WEI)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/QFIN.html"
	rel="suggest" title="QFIN,360 Finance, Inc.,360金融">360金融(QFIN)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/MOGU.html"
	rel="suggest" title="MOGU,Mogu, Inc.,蘑菇街">蘑菇街(MOGU)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/TME.html"
	rel="suggest" title="TME,Tencent Music Entertainment Group,腾讯音乐">腾讯音乐(TME)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/MTC.html"
	rel="suggest" title="MTC,MMTec, Inc.,美美证券">美美证券(MTC)</a>
<div class="clear"></div>
</div>
<div class="col_div"><label>106家在美上市科技类知名公司:</label>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/MSFT.html"
	rel="suggest" title="MSFT,Microsoft Corp.,微软公司">微软(MSFT)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/GOOG.html"
	rel="suggest" title="GOOG,Alphabet, Inc.,谷歌公司">谷歌(GOOG)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/EBAY.html"
	rel="suggest" title="EBAY,eBay, Inc.,eBay">eBay(EBAY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/AMZN.html"
	rel="suggest" title="AMZN,Amazon.com, Inc.,亚马逊公司">亚马逊(AMZN)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/INTC.html"
	rel="suggest" title="INTC,Intel Corp.,英特尔公司">英特尔(INTC)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/AAPL.html"
	rel="suggest" title="AAPL,Apple, Inc.,苹果公司">苹果(AAPL)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/NOK.html"
	rel="suggest" title="NOK,Nokia Oyj,诺基亚公司">诺基亚(NOK)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ERIC.html"
	rel="suggest" title="ERIC,Telefonaktiebolaget LM Ericsson,爱立信电信公司">爱立信(ERIC)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ADBE.html"
	rel="suggest" title="ADBE,Adobe, Inc.,Adobe">Adobe(ADBE)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/IBM.html"
	rel="suggest" title="IBM,International Business Machines Corp.,国际商业机器有限公司">IBM(IBM)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/TXN.html"
	rel="suggest" title="TXN,Texas Instruments Incorporated,德州仪器">德州仪器(TXN)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CSCO.html"
	rel="suggest" title="CSCO,Cisco Systems, Inc.,思科系统公司">思科(CSCO)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/T.html"
	rel="suggest" title="T,AT&T, Inc.,美国电话电报公司">美国电话(T)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/VOD.html"
	rel="suggest" title="VOD,Vodafone Group Plc,沃达丰公司">沃达丰(VOD)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/STX.html"
	rel="suggest" title="STX,Seagate Technology Plc,希捷科技公司">希捷(STX)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ORCL.html"
	rel="suggest" title="ORCL,Oracle Corp.,甲骨文公司">甲骨文(ORCL)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/QCOM.html"
	rel="suggest" title="QCOM,QUALCOMM, Inc.,高通公司">高通(QCOM)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/S.html"
	rel="suggest" title="S,Sprint Corp.,斯普林特奈科斯特公司">斯普林特(S)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/EMC.html"
	rel="suggest" title="EMC,Dell EMC, Inc.,美国易安信公司">易安信(EMC)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CTSH.html"
	rel="suggest" title="CTSH,Cognizant Technology Solutions Corp.,高知特信息技术">高知特(CTSH)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/VZ.html"
	rel="suggest" title="VZ,Verizon Communications, Inc.,威讯通信公司">威讯(VZ)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ATI.html"
	rel="suggest" title="ATI,Allegheny Technologies, Inc.,阿利根尼技术公司">阿利根尼(ATI)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/HPQ.html"
	rel="suggest" title="HPQ,HP, Inc.,惠普">惠普(HPQ)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/AMD.html"
	rel="suggest" title="AMD,Advanced Micro Devices, Inc.,美国超微公司">AMD(AMD)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/NVDA.html"
	rel="suggest" title="NVDA,NVIDIA Corp.,英伟达公司">英伟达(NVDA)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/SAP.html"
	rel="suggest" title="SAP,SAP SE,SAP">SAP(SAP)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/AKAM.html"
	rel="suggest" title="AKAM,Akamai Technologies, Inc.,阿克迈技术">阿克迈(AKAM)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/MSI.html"
	rel="suggest" title="MSI,Motorola Solutions, Inc.,摩托罗拉系统公司">摩托罗拉(MSI)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/MMI.html"
	rel="suggest" title="MMI,Marcus & Millichap, Inc.,摩托罗拉移动控股公司">摩移动(MMI)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/YNDX.html"
	rel="suggest" title="YNDX,Yandex NV,Yandex">Yandex(YNDX)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/GRPN.html"
	rel="suggest" title="GRPN,Groupon, Inc.,GroupOn">GroupOn(GRPN)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ZNGA.html"
	rel="suggest" title="ZNGA,Zynga, Inc.,Zynga公司">Zynga(ZNGA)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/FB.html"
	rel="suggest" title="FB,Facebook, Inc.,Facebook">Facebook(FB)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/SNE.html"
	rel="suggest" title="SNE,Sony Corp.,索尼">索尼(SNE)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/MON.html"
	rel="suggest" title="MON,Monsanto Co.,孟山都公司">孟山都(MON)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ACN.html"
	rel="suggest" title="ACN,Accenture Plc,埃森哲公司">埃森哲(ACN)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/NFLX.html"
	rel="suggest" title="NFLX,Netflix, Inc.,奈飞公司">奈飞(NFLX)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/VMW.html"
	rel="suggest" title="VMW,VMware, Inc.,威睿公司">威睿(VMW)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/TYL.html"
	rel="suggest" title="TYL,Tyler Technologies, Inc.,泰勒科技公司">泰勒科技(TYL)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/MXIM.html"
	rel="suggest" title="MXIM,Maxim Integrated Products, Inc.,美信集成产品公司">美信集成(MXIM)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/JNPR.html"
	rel="suggest" title="JNPR,Juniper Networks, Inc.,瞻博网络公司">瞻博网络(JNPR)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CTXS.html"
	rel="suggest" title="CTXS,Citrix Systems, Inc.,美国思杰公司">思杰系统(CTXS)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/EQIX.html"
	rel="suggest" title="EQIX,Equinix, Inc.,易昆尼克斯公司">易昆尼克(EQIX)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/RHT.html"
	rel="suggest" title="RHT,Red Hat, Inc.,红帽公司">红帽(RHT)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/TDS.html"
	rel="suggest" title="TDS,Telephone & Data Systems, Inc.,美国电话数据系统公司">电话数据(TDS)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/KLAC.html"
	rel="suggest" title="KLAC,KLA-Tencor Corp.,科磊公司">科磊(KLAC)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/YELP.html"
	rel="suggest" title="YELP,Yelp, Inc.,Yelp">Yelp(YELP)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/XLNX.html"
	rel="suggest" title="XLNX,Xilinx, Inc.,赛灵思公司">赛灵思(XLNX)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CRM.html"
	rel="suggest" title="CRM,salesforce.com, inc.,赛富时公司">赛富时(CRM)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/JBL.html"
	rel="suggest" title="JBL,Jabil, Inc.,捷普集团">捷普(JBL)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/NTAP.html"
	rel="suggest" title="NTAP,NetApp, Inc.,美国网存公司">美国网存(NTAP)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/SYMC.html"
	rel="suggest" title="SYMC,Symantec Corp.,赛门铁克公司">赛门铁克(SYMC)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ACLS.html"
	rel="suggest" title="ACLS,Axcelis Technologies, Inc.,Axcelis科技设计公司">Axcelis(ACLS)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/GLW.html"
	rel="suggest" title="GLW,Corning, Inc.,康宁公司">康宁(GLW)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ACM.html"
	rel="suggest" title="ACM,AECOM,AECOM">AECOM(ACM)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/TER.html"
	rel="suggest" title="TER,Teradyne, Inc.,泰瑞达公司">泰瑞达(TER)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/TRIP.html"
	rel="suggest" title="TRIP,TripAdvisor, Inc.,猫途鹰">猫途鹰(TRIP)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/NOW.html"
	rel="suggest" title="NOW,ServiceNow, Inc.,现在服务公司">现在服务(NOW)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/VRSN.html"
	rel="suggest" title="VRSN,VeriSign, Inc.,威瑞信公司">威瑞信(VRSN)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/DCM.html"
	rel="suggest" title="DCM,NTT DoCoMo, Inc.,NTT道康姆公司">NTT道康(DCM)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/MCHP.html"
	rel="suggest" title="MCHP,Microchip Technology, Inc.,微芯科技公司">微芯科技(MCHP)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ADSK.html"
	rel="suggest" title="ADSK,Autodesk, Inc.,欧特克公司">欧特克(ADSK)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/AZPN.html"
	rel="suggest" title="AZPN,Aspen Technology, Inc.,艾斯本科技公司">艾斯本(AZPN)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ARRS.html"
	rel="suggest" title="ARRS,ARRIS International Plc,艾利斯集团">艾利斯(ARRS)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/NUAN.html"
	rel="suggest" title="NUAN,Nuance Communications, Inc.,微妙通讯公司">微妙通讯(NUAN)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ACXM.html"
	rel="suggest" title="ACXM,Acxiom Holdings, Inc.,阿克西鄂姆公司">阿克西鄂(ACXM)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ASCMA.html"
	rel="suggest" title="ASCMA,Ascent Capital Group, Inc.,ASCENT资本集团">ASCENT资(ASCMA)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/FFIV.html"
	rel="suggest" title="FFIV,F5 Networks, Inc.,F5网络公司">F5网络(FFIV)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/BRCD.html"
	rel="suggest" title="BRCD,Brocade Communications Systems LLC,博科通信">博科(BRCD)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CERN.html"
	rel="suggest" title="CERN,Cerner Corp.,美国塞纳公司">美国塞纳(CERN)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/EA.html"
	rel="suggest" title="EA,Electronic Arts, Inc.,电子艺界公司">电子艺界(EA)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/INTU.html"
	rel="suggest" title="INTU,Intuit, Inc.,财捷集团">财捷(INTU)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ATVI.html"
	rel="suggest" title="ATVI,Activision Blizzard, Inc.,动视暴雪公司">动视暴雪(ATVI)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/SATS.html"
	rel="suggest" title="SATS,EchoStar Corp.,回声星通信">回声星通(SATS)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CREE.html"
	rel="suggest" title="CREE,Cree, Inc.,克里科技公司">克里科技(CREE)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/TRMB.html"
	rel="suggest" title="TRMB,Trimble, Inc.,天宝导航公司">天宝导航(TRMB)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/SSYS.html"
	rel="suggest" title="SSYS,Stratasys Ltd.,Stratasys">Stratasy(SSYS)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ATNI.html"
	rel="suggest" title="ATNI,ATN International, Inc.,大西洋电讯网络公司">大西洋电(ATNI)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/AMAT.html"
	rel="suggest" title="AMAT,Applied Materials, Inc.,应用材料公司">应用材料(AMAT)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/WDAY.html"
	rel="suggest" title="WDAY,Workday, Inc.,工作日公司">工时公司(WDAY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/IGT.html"
	rel="suggest" title="IGT,International Game Technology Plc,国际游戏科技公司">游戏科技(IGT)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/TDC.html"
	rel="suggest" title="TDC,Teradata Corp.,天睿公司">天睿(TDC)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/WDC.html"
	rel="suggest" title="WDC,Western Digital Corp.,西部数据公司">西部数据(WDC)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/SYNT.html"
	rel="suggest" title="SYNT,Syntel, Inc.,Syntel公司">Syntel(SYNT)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/LRCX.html"
	rel="suggest" title="LRCX,Lam Research Corp.,林氏研究公司">林氏研究(LRCX)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CA.html"
	rel="suggest" title="CA,CA, Inc.,联合电脑公司">联合电脑(CA)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/FTNT.html"
	rel="suggest" title="FTNT,Fortinet, Inc.,飞塔信息公司">飞塔信息(FTNT)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CDNS.html"
	rel="suggest" title="CDNS,Cadence Design Systems, Inc.,铿腾电子科技有限公司">铿腾电子(CDNS)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/DDD.html"
	rel="suggest" title="DDD,3D Systems Corp.,3D系统公司">3D系统(DDD)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CVLT.html"
	rel="suggest" title="CVLT,Commvault Systems, Inc.,康沃系统公司">康沃系统(CVLT)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ADI.html"
	rel="suggest" title="ADI,Analog Devices, Inc.,亚德诺半导体">亚德诺(ADI)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/FTR.html"
	rel="suggest" title="FTR,Frontier Communications Corp.,边际通信公司">边际通信(FTR)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/UBNT.html"
	rel="suggest" title="UBNT,Ubiquiti Networks, Inc.,厄比奎蒂网络公司">厄比奎蒂(UBNT)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/XRX.html"
	rel="suggest" title="XRX,Xerox Corp.,施乐公司">施乐(XRX)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ARW.html"
	rel="suggest" title="ARW,Arrow Electronics, Inc.,艾睿电子公司">艾睿(ARW)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/MDSO.html"
	rel="suggest" title="MDSO,Medidata Solutions, Inc.,Medidata解决方案公司">Medidata(MDSO)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/APH.html"
	rel="suggest" title="APH,Amphenol Corp.,安费诺集团">安费诺(APH)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/MU.html"
	rel="suggest" title="MU,Micron Technology, Inc.,美光科技公司">美光(MU)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/DLB.html"
	rel="suggest" title="DLB,Dolby Laboratories, Inc.,杜比实验室公司">杜比(DLB)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/BBRY.html"
	rel="suggest" title="BBRY,Research In Motion Limited,黑莓公司">黑莓(BBRY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/TWTR.html"
	rel="suggest" title="TWTR,Twitter, Inc.,Twitter">推特(TWTR)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/NTDOY.html"
	rel="suggest" title="NTDOY,Nintendo Co., Ltd.,任天堂">任天堂(NTDOY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/PYPL.html"
	rel="suggest" title="PYPL,PayPal Holdings, Inc.,PayPal">贝宝(PYPL)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/SNAP.html"
	rel="suggest" title="SNAP,Snap, Inc.,Snap">Snap(SNAP)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/OPRA.html"
	rel="suggest" title="OPRA,Opera Ltd.,欧朋公司">欧朋(OPRA)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/SONO.html"
	rel="suggest" title="SONO,Sonos, Inc.,搜诺思公司">搜诺思(SONO)</a>
<div class="clear"></div>
</div>
<div class="col_div"><label>39家在美上市金融类知名公司:</label>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/GS.html"
	rel="suggest" title="GS,The Goldman Sachs Group, Inc.,高盛集团">高盛(GS)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/C.html"
	rel="suggest" title="C,Citigroup, Inc.,花旗集团">花旗(C)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/BAC.html"
	rel="suggest" title="BAC,Bank of America Corp.,美国银行公司">美国银行(BAC)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/JPM.html"
	rel="suggest" title="JPM,JPMorgan Chase & Co.,摩根大通公司">摩根大通(JPM)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/MS.html"
	rel="suggest" title="MS,Morgan Stanley,摩根士丹利">摩根士丹(MS)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/AIG.html"
	rel="suggest" title="AIG,American International Group, Inc.,美国国际集团">美国国际(AIG)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/MET.html"
	rel="suggest" title="MET,MetLife, Inc.,大都会人寿公司">大都会人(MET)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/MA.html"
	rel="suggest" title="MA,Mastercard, Inc.,万事达卡公司">万事达(MA)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/V.html"
	rel="suggest" title="V,Visa, Inc.,维萨卡公司">维萨卡(V)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/WFC.html"
	rel="suggest" title="WFC,Wells Fargo & Co.,富国银行">富国银行(WFC)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/BRK.A.html"
	rel="suggest" title="BRK.A,Berkshire Hathaway, Inc.,伯克希尔-哈撒韦">伯克希尔(BRK.A)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/RF.html"
	rel="suggest" title="RF,Regions Financial Corp.,地区金融公司">地区金融(RF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/BK.html"
	rel="suggest" title="BK,The Bank of New York Mellon Corp.,纽约梅隆银行公司">纽约梅隆(BK)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/AXP.html"
	rel="suggest" title="AXP,American Express Co.,美国运通公司">美国运通(AXP)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CI.html"
	rel="suggest" title="CI,Cigna Corp.,信诺">信诺(CI)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/USB.html"
	rel="suggest" title="USB,U.S. Bancorp,美国合众银行">合众银行(USB)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/COF.html"
	rel="suggest" title="COF,Capital One Financial Corp.,第一资本金融公司">第一资本(COF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ALL.html"
	rel="suggest" title="ALL,The Allstate Corp.,美国好事达保险公司">好事达(ALL)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/HIG.html"
	rel="suggest" title="HIG,The Hartford Financial Services Group, Inc.,美国哈特福德金融服务公司">哈特福德(HIG)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/TRV.html"
	rel="suggest" title="TRV,The Travelers Cos., Inc.,旅行者财产险集团">旅行者财(TRV)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/MTU.html"
	rel="suggest" title="MTU,Mitsubishi UFJ Financial Group, Inc.,三菱日联金融集团">三菱日联(MTU)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/MFG.html"
	rel="suggest" title="MFG,Mizuho Financial Group, Inc.,瑞穗金融集团">瑞穗金融(MFG)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/SMFG.html"
	rel="suggest" title="SMFG,Sumitomo Mitsui Financial Group, Inc.,三井住友金融集团">三井住友(SMFG)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/NMR.html"
	rel="suggest" title="NMR,Nomura Holdings, Inc.,野村控股">野村控股(NMR)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/SPG.html"
	rel="suggest" title="SPG,Simon Property Group, Inc.,西蒙地产集团">西蒙地产(SPG)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/AF.html"
	rel="suggest" title="AF,Astoria Financial Corp.,道夫金融公司">道夫金融(AF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/FHN.html"
	rel="suggest" title="FHN,First Horizon National Corp. (Tennessee),第一线国民银行">一线国民(FHN)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ARR.html"
	rel="suggest" title="ARR,ARMOUR Residential REIT, Inc.,ARMOUR住宅房地产资金信托">ARMOUR住(ARR)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/BX.html"
	rel="suggest" title="BX,The Blackstone Group LP,黑石集团">黑石(BX)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/GCAP.html"
	rel="suggest" title="GCAP,GAIN Capital Holdings, Inc.,嘉盛集团">嘉盛集团(GCAP)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ATLO.html"
	rel="suggest" title="ATLO,Ames National Corp.,艾姆斯银行控股">艾姆斯(ATLO)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ZION.html"
	rel="suggest" title="ZION,Zions Bancorporation NA,锡安银行">锡安银行(ZION)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CMA.html"
	rel="suggest" title="CMA,Comerica, Inc.,联信银行">联信(CMA)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ASFI.html"
	rel="suggest" title="ASFI,Asta Funding, Inc.,阿斯塔资金公司">阿斯塔(ASFI)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/FII.html"
	rel="suggest" title="FII,Federated Investors, Inc.,联邦投资公司">联邦投资(FII)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CINF.html"
	rel="suggest" title="CINF,Cincinnati Financial Corp.,辛辛那提金融公司">辛市金融(CINF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/AMNB.html"
	rel="suggest" title="AMNB,American National Bankshares, Inc. (Virginia),美国国家银行股份公司">美国国家(AMNB)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/BRK.B.html"
	rel="suggest" title="BRK.B,Berkshire Hathaway, Inc.,伯克希尔-哈撒韦B">伯克希尔(BRK.B)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/LC.html"
	rel="suggest" title="LC,LendingClub Corp.,LendingClub">LendingC(LC)</a>
<div class="clear"></div>
</div>
<div class="col_div"><label>42家在美上市医药、食品类知名公司:</label>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/PEP.html"
	rel="suggest" title="PEP,PepsiCo, Inc.,百事可乐公司">百事可乐(PEP)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/JNJ.html"
	rel="suggest" title="JNJ,Johnson & Johnson,强生公司">强生(JNJ)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/PFE.html"
	rel="suggest" title="PFE,Pfizer Inc.,辉瑞制药公司">辉瑞(PFE)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/KO.html"
	rel="suggest" title="KO,The Coca-Cola Co.,可口可乐公司">可口可乐(KO)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/MDT.html"
	rel="suggest" title="MDT,Medtronic Plc,美敦力公司">美敦力(MDT)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/BMY.html"
	rel="suggest" title="BMY,Bristol-Myers Squibb Co.,百时美施贵宝公司">施贵宝(BMY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/MO.html"
	rel="suggest" title="MO,Altria Group, Inc.,奥驰亚集团公司">奥驰亚(MO)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/UNH.html"
	rel="suggest" title="UNH,UnitedHealth Group, Inc.,美国联合健康集团">联合健康(UNH)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/MRK.html"
	rel="suggest" title="MRK,Merck & Co., Inc.,默沙东集团">默沙东(MRK)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/AMGN.html"
	rel="suggest" title="AMGN,Amgen, Inc.,安进公司">安进(AMGN)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CVS.html"
	rel="suggest" title="CVS,CVS Health Corp.,CVS健保公司">CVS健保(CVS)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/BAX.html"
	rel="suggest" title="BAX,Baxter International, Inc.,百特国际有限公司">百特国际(BAX)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ABT.html"
	rel="suggest" title="ABT,Abbott Laboratories,雅培公司">雅培(ABT)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/SBUX.html"
	rel="suggest" title="SBUX,Starbucks Corp.,星巴克公司">星巴克(SBUX)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CPB.html"
	rel="suggest" title="CPB,Campbell Soup Co.,康宝浓汤公司">康宝浓汤(CPB)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/MCD.html"
	rel="suggest" title="MCD,McDonald's Corp.,麦当劳公司">麦当劳(MCD)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/RHHBY.html"
	rel="suggest" title="RHHBY,Roche Holding AG,罗氏制药">罗氏制药(RHHBY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/GSK.html"
	rel="suggest" title="GSK,GlaxoSmithKline Plc,葛兰素史克公司">葛兰素史(GSK)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/YUM.html"
	rel="suggest" title="YUM,Yum! Brands, Inc.,百胜餐饮集团">百胜餐饮(YUM)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/GILD.html"
	rel="suggest" title="GILD,Gilead Sciences, Inc.,吉利德科学公司">吉利德(GILD)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/MDLZ.html"
	rel="suggest" title="MDLZ,Mondelez International, Inc.,亿滋国际公司">亿滋(MDLZ)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ESRX.html"
	rel="suggest" title="ESRX,Express Scripts Holding Co.,快捷药方公司">快捷药方(ESRX)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ATRI.html"
	rel="suggest" title="ATRI,Atrion Corp.,Atrion公司公司">Atrion公(ATRI)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ARNA.html"
	rel="suggest" title="ARNA,Arena Pharmaceuticals, Inc.,阿里那制药公司">阿里那(ARNA)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ACHN.html"
	rel="suggest" title="ACHN,Achillion Pharmaceuticals, Inc.,艾琪尔顿制药公司">艾琪尔顿(ACHN)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ATEC.html"
	rel="suggest" title="ATEC,Alphatec Holdings, Inc.,ALPHATEC控股公司">ALPHATEC(ATEC)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ATHN.html"
	rel="suggest" title="ATHN,athenahealth, Inc.,雅典娜保健公司">雅典娜(ATHN)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ACUR.html"
	rel="suggest" title="ACUR,Acura Pharmaceuticals, Inc.,讴歌制药公司">讴歌制药(ACUR)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ATRS.html"
	rel="suggest" title="ATRS,Antares Pharma, Inc.,安塔尔制药公司">安塔尔(ATRS)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/AFAM.html"
	rel="suggest" title="AFAM,Almost Family, Inc.,阿莫斯特家庭保健">阿莫斯特(AFAM)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/AEGR.html"
	rel="suggest" title="AEGR,Aegerion Pharmaceuticals, Inc.,Aegerion制药">Aegerion(AEGR)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ABAX.html"
	rel="suggest" title="ABAX,ABAXIS, Inc.,爱贝斯公司">爱贝斯公(ABAX)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ARRY.html"
	rel="suggest" title="ARRY,Array BioPharma, Inc.,Array生物制药">Array生(ARRY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/AVEO.html"
	rel="suggest" title="AVEO,AVEO Pharmaceuticals, Inc.,AVEO制药公司">AVEO制药(AVEO)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ARIA.html"
	rel="suggest" title="ARIA,ARIAD Pharmaceuticals, Inc.,阿里阿德制药公司">阿里阿德(ARIA)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/HLF.html"
	rel="suggest" title="HLF,Herbalife Nutrition Ltd.,康宝莱国际公司">康宝莱(HLF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/VRX.html"
	rel="suggest" title="VRX,Valeant Pharmaceuticals International, Inc.,瓦伦特国际制药公司">瓦伦特制(VRX)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/AGN.html"
	rel="suggest" title="AGN,Allergan Plc,艾尔建制药公司">艾尔建(AGN)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ATRC.html"
	rel="suggest" title="ATRC,AtriCure, Inc.,AtriCure公司">AtriCure(ATRC)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/BF.B.html"
	rel="suggest" title="BF.B,Brown-Forman Corp.,布朗霍文集团">布朗霍文(BF.B)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/BF.A.html"
	rel="suggest" title="BF.A,Brown-Forman Corp.,布朗霍文集团">布朗霍文(BF.A)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/YUMC.html"
	rel="suggest" title="YUMC,Yum China Holdings, Inc.,百胜中国控股有限公司">百胜中国(YUMC)</a>
<div class="clear"></div>
</div>
<div class="col_div"><label>10家在美上市媒体类知名公司:</label>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/NWS.html"
	rel="suggest" title="NWS,News Corp.,新闻集团">新闻集团(NWS)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/TRI.html"
	rel="suggest" title="TRI,Thomson Reuters Corp.,汤姆逊-路透">汤姆逊-(TRI)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/TWX.html"
	rel="suggest" title="TWX,Time Warner, Inc.,时代华纳公司">时代华纳(TWX)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/DIS.html"
	rel="suggest" title="DIS,The Walt Disney Co.,沃特迪士尼公司">迪士尼(DIS)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CCU.html"
	rel="suggest" title="CCU,Cia Cervecerias Unidas SA,清晰频道">清晰频道(CCU)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CBS.html"
	rel="suggest" title="CBS,CBS Corp.,哥伦比亚广播公司">哥伦比亚(CBS)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CMCSA.html"
	rel="suggest" title="CMCSA,Comcast Corp.,康卡斯特公司">康卡斯特(CMCSA)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/NWSA.html"
	rel="suggest" title="NWSA,News Corp.,新闻集团">新闻集团(NWSA)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/SIRI.html"
	rel="suggest" title="SIRI,Sirius XM Holdings, Inc.,天狼星XM">天狼星XM(SIRI)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/P.html"
	rel="suggest" title="P,Pandora Media, Inc.,潘多拉媒体">潘多拉媒(P)</a>
<div class="clear"></div>
</div>
<div class="col_div"><label>36家在美上市汽车、能源类知名公司:</label>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/GM.html"
	rel="suggest" title="GM,General Motors Co.,通用汽车公司">通用汽车(GM)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/F.html"
	rel="suggest" title="F,Ford Motor Co.,福特汽车公司">福特汽车(F)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/XOM.html"
	rel="suggest" title="XOM,Exxon Mobil Corp.,埃克森美孚公司">埃克森美(XOM)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/COP.html"
	rel="suggest" title="COP,ConocoPhillips,康菲石油公司">康菲石油(COP)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/RIO.html"
	rel="suggest" title="RIO,Rio Tinto Plc,力拓公司">力拓(RIO)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CAT.html"
	rel="suggest" title="CAT,Caterpillar, Inc.,卡特彼勒公司">卡特彼勒(CAT)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/DUK.html"
	rel="suggest" title="DUK,Duke Energy Corp.,杜克能源公司">杜克能源(DUK)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/HAL.html"
	rel="suggest" title="HAL,Halliburton Co.,哈里伯顿公司">哈里伯顿(HAL)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/SLB.html"
	rel="suggest" title="SLB,Schlumberger NV,斯伦贝谢公司">斯伦贝谢(SLB)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/AES.html"
	rel="suggest" title="AES,The AES Corp.,爱依斯电力">爱依斯(AES)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/WMB.html"
	rel="suggest" title="WMB,The Williams Cos., Inc.,威廉姆斯公司">威廉姆斯(WMB)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/BHI.html"
	rel="suggest" title="BHI,Baker Hughes Incorporated,贝克休斯公司">贝克休斯(BHI)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/SO.html"
	rel="suggest" title="SO,The Southern Co.,美国南方公司">美国南方(SO)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ETR.html"
	rel="suggest" title="ETR,Entergy Corp.,安特吉公司">安特吉(ETR)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/EXC.html"
	rel="suggest" title="EXC,Exelon Corp.,艾斯能公司">艾斯能(EXC)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/AEP.html"
	rel="suggest" title="AEP,American Electric Power Co., Inc.,美国电力公司">美国电力(AEP)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CVX.html"
	rel="suggest" title="CVX,Chevron Corp.,雪佛龙公司">雪佛龙(CVX)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/BP.html"
	rel="suggest" title="BP,BP Plc,英国石油公司">英国石油(BP)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/HMC.html"
	rel="suggest" title="HMC,Honda Motor Co., Ltd.,本田汽车公司">本田汽车(HMC)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/TM.html"
	rel="suggest" title="TM,Toyota Motor Corp.,丰田汽车公司">丰田汽车(TM)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/UNP.html"
	rel="suggest" title="UNP,Union Pacific Corp.,联合太平洋公司">联合太平(UNP)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ACIM.html"
	rel="suggest" title="ACIM,SPDR MSCI ACWI IMI ETF,拱煤公司">拱煤公司(ACIM)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/HP.html"
	rel="suggest" title="HP,Helmerich & Payne, Inc.,赫佩公司">赫佩公司(HP)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/TSLA.html"
	rel="suggest" title="TSLA,Tesla, Inc.,特斯拉汽车">特斯拉(TSLA)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/AVA.html"
	rel="suggest" title="AVA,Avista Corp.,阿维斯塔公司">阿维斯塔(AVA)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/AVAV.html"
	rel="suggest" title="AVAV,AeroVironment, Inc.,AeroVironment公司">AeroViro(AVAV)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/AEIS.html"
	rel="suggest" title="AEIS,Advanced Energy Industries, Inc.,先进能源工业公司">先进能源(AEIS)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/AAON.html"
	rel="suggest" title="AAON,AAON, Inc.,艾伦建材">艾伦建材(AAON)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ARTNA.html"
	rel="suggest" title="ARTNA,Artesian Resources Corp.,自流资源公司">自流资源(ARTNA)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ATO.html"
	rel="suggest" title="ATO,Atmos Energy Corp.,ATMOS能源公司">ATMOS能(ATO)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ADGE.html"
	rel="suggest" title="ADGE,American DG Energy, Inc.,美国DG能源公司">美国DG能(ADGE)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/SRE.html"
	rel="suggest" title="SRE,Sempra Energy,桑普拉能源公司">桑普拉能(SRE)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/RDS.A.html"
	rel="suggest" title="RDS.A,Royal Dutch Shell Plc,荷兰皇家壳牌石油公司">壳牌石油(RDS.A)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/RDS.B.html"
	rel="suggest" title="RDS.B,Royal Dutch Shell Plc,荷兰皇家壳牌石油公司">壳牌石油(RDS.B)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/FCAU.html"
	rel="suggest" title="FCAU,Fiat Chrysler Automobiles NV,菲亚特-克莱斯勒汽车公司">菲亚特克(FCAU)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/RACE.html"
	rel="suggest" title="RACE,Ferrari NV,法拉利汽车公司">法拉利(RACE)</a>
<div class="clear"></div>
</div>
<div class="col_div"><label>48家在美上市制造、零售类知名公司:</label>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/GE.html"
	rel="suggest" title="GE,General Electric Co.,通用电气公司">通用电气(GE)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/NKE.html"
	rel="suggest" title="NKE,NIKE, Inc.,耐克公司">耐克(NKE)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/WMT.html"
	rel="suggest" title="WMT,Walmart, Inc.,沃尔玛公司">沃尔玛(WMT)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/PG.html"
	rel="suggest" title="PG,Procter & Gamble Co.,宝洁公司">宝洁(PG)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/BA.html"
	rel="suggest" title="BA,The Boeing Co.,波音公司">波音(BA)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/GD.html"
	rel="suggest" title="GD,General Dynamics Corp.,通用动力">通用动力(GD)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/HON.html"
	rel="suggest" title="HON,Honeywell International, Inc.,霍尼韦尔国际公司">霍尼韦尔(HON)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/RTN.html"
	rel="suggest" title="RTN,Raytheon Co.,雷神公司">雷神(RTN)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/MMM.html"
	rel="suggest" title="MMM,3M Co.,明尼苏达矿业制造公司">3M(MMM)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/EL.html"
	rel="suggest" title="EL,The Est茅e Lauder Companies, Inc.,雅诗兰黛公司">雅诗兰黛(EL)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/AVP.html"
	rel="suggest" title="AVP,Avon Products, Inc.,美国雅芳产品有限公司">雅芳(AVP)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CL.html"
	rel="suggest" title="CL,Colgate-Palmolive Co.,高露洁棕榄有限公司">高露洁(CL)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/HD.html"
	rel="suggest" title="HD,The Home Depot, Inc.,家得宝公司">家得宝(HD)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/FDX.html"
	rel="suggest" title="FDX,FedEx Corp.,联邦快递集团">联邦快递(FDX)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ROK.html"
	rel="suggest" title="ROK,Rockwell Automation, Inc.,罗克韦尔公司">罗克韦尔(ROK)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/APOL.html"
	rel="suggest" title="APOL,Apollo Education Group, Inc.,阿波罗集团">阿波罗(APOL)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/IP.html"
	rel="suggest" title="IP,International Paper Co.,国际纸业公司">国际纸业(IP)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/TGT.html"
	rel="suggest" title="TGT,Target Corp.,塔吉特公司">塔吉特(TGT)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/NSC.html"
	rel="suggest" title="NSC,Norfolk Southern Corp.,诺福克南方公司">诺福克南(NSC)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/WY.html"
	rel="suggest" title="WY,Weyerhaeuser Co.,惠好公司">惠好(WY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/TYC.html"
	rel="suggest" title="TYC,Tyco International Ltd.,泰科国际有限公司">泰科国际(TYC)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/UTX.html"
	rel="suggest" title="UTX,United Technologies Corp.,联合技术公司">联合技术(UTX)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/AA.html"
	rel="suggest" title="AA,Alcoa Corp.,美国铝业公司">美国铝业(AA)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/HLT.html"
	rel="suggest" title="HLT,Hilton Worldwide Holdings, Inc.,希尔顿酒店">希尔顿酒(HLT)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/B.html"
	rel="suggest" title="B,Barnes Group, Inc.,巴尼斯集团">巴恩斯(B)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CAJ.html"
	rel="suggest" title="CAJ,Canon, Inc.,佳能公司">佳能(CAJ)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/KYO.html"
	rel="suggest" title="KYO,Kyocera Corp.,日本京瓷">日本京瓷(KYO)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/COST.html"
	rel="suggest" title="COST,Costco Wholesale Corp.,好市多公司">好市多(COST)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ASNA.html"
	rel="suggest" title="ASNA,Ascena Retail Group, Inc.,Ascena零售集团">Ascena(ASNA)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/UA.html"
	rel="suggest" title="UA,Under Armour, Inc.,安德玛公司">安德玛(UA)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ATRO.html"
	rel="suggest" title="ATRO,Astronics Corp.,Astronics公司">Astronic(ATRO)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ACW.html"
	rel="suggest" title="ACW,Accuride Corp.,雅固拉公司">雅固拉(ACW)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/BIG.html"
	rel="suggest" title="BIG,Big Lots, Inc.,必乐透公司">必乐透(BIG)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/HOG.html"
	rel="suggest" title="HOG,Harley-Davidson, Inc.,哈雷戴维森公司">哈雷摩托(HOG)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ATU.html"
	rel="suggest" title="ATU,Actuant Corp.,实用动力集团">实用动力(ATU)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ASTE.html"
	rel="suggest" title="ASTE,Astec Industries, Inc.,Astec实业公司">Astec实(ASTE)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/AVD.html"
	rel="suggest" title="AVD,American Vanguard Corp.,美国先锋公司">美国先锋(AVD)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ASEI.html"
	rel="suggest" title="ASEI,American Science & Engineering, Inc.,美国科学工程公司">美国科工(ASEI)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ATW.html"
	rel="suggest" title="ATW,Atwood Oceanics, Inc.,Atwood海洋工程公司">Atwood海(ATW)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ADC.html"
	rel="suggest" title="ADC,Agree Realty Corp.,美国同意房地产公司">同意房产(ADC)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/AEO.html"
	rel="suggest" title="AEO,American Eagle Outfitters, Inc.,美国鹰君服饰公司">美鹰服饰(AEO)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ASH.html"
	rel="suggest" title="ASH,Ashland Global Holdings, Inc.,亚什兰公司">亚什兰(ASH)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ANF.html"
	rel="suggest" title="ANF,Abercrombie & Fitch Co.,爱芬奇公司">爱芬奇(ANF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ACRE.html"
	rel="suggest" title="ACRE,Ares Commercial Real Estate Corp.,战神商业房地产公司">战神房产(ACRE)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/JCP.html"
	rel="suggest" title="JCP,J. C. Penney Co., Inc.,彭尼公司">彭尼百货(JCP)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ATR.html"
	rel="suggest" title="ATR,AptarGroup, Inc.,Aptar集团公司">Aptar(ATR)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/HAS.html"
	rel="suggest" title="HAS,Hasbro, Inc.,孩之宝公司">孩之宝(HAS)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/GOOS.html"
	rel="suggest" title="GOOS,Canada Goose Holdings, Inc.,加拿大鹅控股公司">加拿大鹅(GOOS)</a>
<div class="clear"></div>
</div>
<div class="col_div"><label>25家在美知名ETF:</label>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/ERX.html"
	rel="suggest" title="ERX,Direxion Daily Energy Bull 3x Shares,Direxion三倍做多能源股ETF">3X多能源(ERX)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/SPY.html"
	rel="suggest" title="SPY,SPDR S&P 500 ETF Trust,标普500ETF">标普ETF(SPY)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/DIA.html"
	rel="suggest" title="DIA,SPDR Dow Jones Industrial Average ETF Trust,SPDR追踪道指ETF">道指ETF(DIA)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/XLE.html"
	rel="suggest" title="XLE,Energy Select Sector SPDR Fund,SPDR追踪精选能源股指ETF">能源ETF(XLE)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/XLF.html"
	rel="suggest" title="XLF,Financial Select Sector SPDR Fund,SPDR追踪精选金融股指TF">金融ETF(XLF)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/XLV.html"
	rel="suggest" title="XLV,Health Care Select Sector SPDR Fund,SPDR追踪精选健保股指ETF">健保ETF(XLV)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/XLI.html"
	rel="suggest" title="XLI,Industrial Select Sector SPDR Fund,SPDR追踪精选工业股指ETF">工业ETF(XLI)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/XLB.html"
	rel="suggest" title="XLB,Materials Select Sector SPDR Fund,SPDR追踪精选原材料股指ETF">材料ETF(XLB)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/IYZ.html"
	rel="suggest" title="IYZ,iShares U.S. Telecommunications ETF,iShares追踪道琼电信股指ETF">电信ETF(IYZ)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/USO.html"
	rel="suggest" title="USO,United States Oil Fund LP,美国WTI油价ETF">油价ETF(USO)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/UNG.html"
	rel="suggest" title="UNG,United States Natural Gas Fund LP,美国天然气价ETF">气价ETF(UNG)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/GLD.html"
	rel="suggest" title="GLD,SPDR Gold Trust,SPDR追踪金价ETF">金价ETF(GLD)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/FAZ.html"
	rel="suggest" title="FAZ,Direxion Daily Financial Bear 3X Shares,Direxion三倍做空金融股指ETF">3X空金融(FAZ)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/FAS.html"
	rel="suggest" title="FAS,Direxion Daily Financial Bull 3x Shares,Direxion三倍做多金融股指ETF">3X多金融(FAS)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/SDS.html"
	rel="suggest" title="SDS,ProShares UltraShort S&P500,ProShares二倍做空标普500指数ETF">2X空标普(SDS)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/SSO.html"
	rel="suggest" title="SSO,ProShares Ultra S&P500,ProShares二倍做多标普500指数ETF">2X多标普(SSO)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/QQQ.html"
	rel="suggest" title="QQQ,Invesco QQQ Trust,纳指100ETF">纳百ETF(QQQ)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/FXI.html"
	rel="suggest" title="FXI,iShares China Large-Cap ETF,iShares追踪富时25中国股指数ETF">富时中股(FXI)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/FXB.html"
	rel="suggest" title="FXB,Invesco CurrencyShares British Pound Sterling Trust,CurrencyShares追踪英镑币值ETF">英镑ETF(FXB)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/FXE.html"
	rel="suggest" title="FXE,Invesco CurrencyShares Euro Trust,CurrencyShares追踪欧元币值ETF">欧元ETF(FXE)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/YINN.html"
	rel="suggest" title="YINN,Direxion Daily FTSE China Bull 3X Shares,三倍做多A股新华50指数ETF">三倍做多(YINN)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/TVIX.html"
	rel="suggest" title="TVIX,VelocityShares Daily 2x VIX Short-Term ETN,VelocityShares Daily 2x VIX Short-Term ETN">Velocity(TVIX)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/NUGT.html"
	rel="suggest" title="NUGT,Direxion Daily Gold Miners Index Bull 3x Shares,Direxion三倍做多金矿股ETF">3X多金矿(NUGT)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CNXT.html"
	rel="suggest" title="CNXT,VanEck Vectors ChinaAMC SME-ChiNext ETF,MV中创100 ETF">中创100 (CNXT)</a>
<a href="http://stock.finance.sina.com.cn/usstock/quotes/CBON.html"
	rel="suggest" title="CBON,VanEck Vectors ChinaAMC China Bond ETF,中国高级债指数ETF">中国高级(CBON)</a>
	"""

soup = BeautifulSoup(html, features="lxml")
df = pd.DataFrame()
a = soup.select("a")
for link in a:
    # print link.text
    name = re.sub(u"\\(.*?\\)|\\{.*?}|\\[.*?]", "", link.text).strip()
    # name = name.decode('iso-8859-1').encode('windows-1252', 'ignore')
    symbol = re.findall(r'[(](.*?)[)]', link.text)[0].strip()
    # print name, symbol
    df = df.append([{'Name_CN': name, 'Symbol': symbol}], ignore_index=True)
print len(df)
company = store.get_usa_company()
company = pd.merge(company, df, how='left', on=['Symbol'])
print company
from sqlalchemy import create_engine
engine = create_engine('mysql://root:root@127.0.0.1:3306/Stock?charset=utf8')
# company.to_sql('usa_company_1', engine, if_exists='append')
company.to_csv('company.csv', index=False, header=True, encoding='gbk')
