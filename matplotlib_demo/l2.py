#!/usr/bin/env python
# -*- coding: utf-8 -*-
# coding: utf-8
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# https://www.jianshu.com/p/b02ec7dc39dd
from tools import DrawTools


def test0():
    # /Users/a1800101471/.matplotlib
    print('matplotlib的缓存目录：', matplotlib.get_cachedir())
    # /Users/a1800101471/Library/Python/3.7/lib/python/site-packages/matplotlib/mpl-data/matplotlibrc
    print('matplotlib的字体库目录：', matplotlib.matplotlib_fname())
    print('matplotlib的支持字体', plt.rcParams['font.family'])
    print('matplotlib的支持字体', plt.rcParams['font.sans-serif'])
    # plt.rcParams['font.sans-serif'] = ['msyh']

    for f in sorted([f.name for f in matplotlib.font_manager.fontManager.ttflist]):
        print(f)


# 简单线性拟合图
def test1():
    np.random.seed(0)
    x1 = np.linspace(0, 10, 50)
    x2 = [0] * 10 + [1] * 40
    y = 2 * x1 + 5 + np.random.random(50) * 10
    data = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})
    sns.lmplot("x1", "y", data=data, hue="x2", legend=False)
    plt.legend(['类别0', '类别1'])
    plt.show()


def test2():
    df = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/mpg_ggplot2.csv")
    # ['manufacturer', 'model', 'displ', 'year', 'cyl', 'trans', 'drv', 'cty','hwy', 'fl', 'class'],
    # name = ["汽车制造商","型号名称","发动机排量(L)","制造年份","气缸数量","手动/自动","驱动类型","城市里程/加仑","公路里程/加仑","汽油种类","车辆种类"]
    ##驱动类型：4:四轮，f:前轮，r:后轮
    # 能源种类：汽油，柴油，用电等等
    # 车辆种类：皮卡，SUV，小型，midsize中型等等
    # 城市里程/加仑，公路里程/加仑：表示使用没加仑汽油能够跑的英里数，所以这个数值越大代表汽车越节能
    # df = df.loc[df.cyl.isin([4, 8]), :]
    DrawTools.init()
    DrawTools.lmplot(data=df, feature_x='displ', feature_y='hwy', feature_h='cyl', xlim=(1, 7), ylim=(0, 50))
    DrawTools.font_desc(title='按气缸数分组的最佳拟合线散点图', xlabel='发动机排量(L)', ylabel='公路里程/加仑', legends=['气缸数量4', '气缸数量8'])
    plt.show()


def test3():
    df = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/mpg_ggplot2.csv")
    DrawTools.init()
    DrawTools.lmplot_mul(data=df, feature_x='displ', feature_y='hwy', feature_h='cyl', xlim=(1, 7), ylim=(0, 50))
    DrawTools.font_desc(title='按气缸数分组的最佳拟合线散点图', xlabel='发动机排量(L)', ylabel='公路里程/加仑', legends=['气缸数量4', '气缸数量8'])
    plt.show()


def test4():
    data = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/mpg_ggplot2.csv")
    # 画 cty 城市里程/加仑 和 hwy 公路里程/加仑 的 点图
    DrawTools.init()
    DrawTools.scatter(data=data, feature_x='cty', feature_y='hwy')
    DrawTools.font_desc(title='关系图', xlabel='城市里程/加仑', ylabel='公路里程/加仑', legends=None)
    plt.show()


def test5():
    df = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/mpg_ggplot2.csv")
    DrawTools.init()
    fig, ax = plt.subplots(figsize=(12, 8), dpi=80)
    # 用来画抖动图的函数：sns.stripplot
    sns.stripplot(df.cty, df.hwy
                  , jitter=0.25  # 抖动的幅度
                  , size=8, ax=ax
                  , linewidth=.5
                  , palette='Reds'
                  )
    # Decorations
    plt.title('Use jittered plots to avoid overlapping of points', fontsize=22)
    plt.rcParams['font.sans-serif'] = ['Simhei']
    plt.xlabel("气缸数量", fontsize=16)
    plt.ylabel("公路里程/加仑", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()


def test6():
    data = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/mpg_ggplot2.csv")
    DrawTools.init()
    fig, ax = plt.subplots(figsize=(12, 8), dpi=80)
    DrawTools.stripplot(data, 'cty', 'hwy', ax)
    DrawTools.font_desc(ax, 'Use jittered plots to avoid overlapping of points', "气缸数量", "公路里程/加仑")
    plt.show()


# 边缘直方图
def test7():
    # 画 displ 发动机排量,cyl 气缸数量,hwy 公路里程/加仑  cyl是 4和8的
    data = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/mpg_ggplot2.csv")
    DrawTools.init()
    fig = plt.figure(figsize=(16, 10), dpi=80)
    grid = plt.GridSpec(4, 4, hspace=0.5, wspace=0.2)
    ax_main = fig.add_subplot(grid[:-1, :-1])
    ax_right = fig.add_subplot(grid[:-1, -1], xticklabels=[], yticklabels=[])
    ax_bottom = fig.add_subplot(grid[-1, :-1], xticklabels=[], yticklabels=[])
    # displ 发动机排量(L)
    # hwy 公路里程/加仑
    DrawTools.scatter(data=data, feature_x='displ', feature_y='hwy', feature_c='manufacturer', feature_s='cty',
                      ax=ax_main)
    DrawTools.font_desc(ax_main, title='边缘直方图 \n 发动机排量 vs 公路里程/加仑', xlabel='发动机排量(L)', ylabel='公里路程/加仑')
    DrawTools.hist(data, feature_x='hwy', bins=40, vertical=False, color='deeppink', ax=ax_right)
    DrawTools.hist(data, feature_x='displ', bins=40, vertical=True, invert_y=True, color='deeppink',
                   ax=ax_bottom)

    xlabels = ax_main.get_xticks().tolist()  # 将现有的标尺取出来，转化为带一位小数的浮点数
    ax_main.set_xticklabels(xlabels)  # 再将带一位小数的浮点数变成标尺
    plt.show()


# 边缘箱线图
def test8():
    data = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/mpg_ggplot2.csv")
    DrawTools.init()
    fig = plt.figure(figsize=(16, 10), dpi=80)
    grid = plt.GridSpec(4, 4, hspace=0.5, wspace=0.2)
    ax_main = fig.add_subplot(grid[:-1, :-1])
    ax_right = fig.add_subplot(grid[:-1, -1], xticklabels=[], yticklabels=[])
    ax_bottom = fig.add_subplot(grid[-1, :-1], xticklabels=[], yticklabels=[])
    # displ 发动机排量(L)
    # hwy 公路里程/加仑
    DrawTools.scatter(data=data, feature_x='displ', feature_y='hwy', feature_c='manufacturer', feature_s='cty',
                      ax=ax_main, xlim=(1, 7), ylim=(0, 50))
    DrawTools.font_desc(ax_main, title='边缘直方图 \n 发动机排量 vs 公路里程/加仑', xlabel='发动机排量(L)', ylabel='公里路程/加仑')
    # 对右侧和下方绘制箱线图
    DrawTools.boxplot(data, 'hwy', vertical=True, color="red", ax=ax_right)
    DrawTools.boxplot(data, 'displ', vertical=False, color="red", ax=ax_bottom)
    ax_bottom.set(xlabel='')
    ax_right.set(ylabel='')

    xlabels = ax_main.get_xticks().tolist()  # 将现有的标尺取出来，转化为带一位小数的浮点数
    ax_main.set_xticklabels(xlabels)  # 再将带一位小数的浮点数变成标尺
    plt.show()


def test9():
    df = pd.read_csv("https://github.com/selva86/datasets/raw/master/mtcars.csv")
    name = ["英里/加仑", "气缸数量", "排量", "总马力", "驱动轴比", "重量"
        , "1/4英里所用时间", "引擎", "变速器", "前进档数", "化油器数量", "用油是否高效"
        , "汽车", "汽车名称"]
    df.columns = name
    DrawTools.init()
    DrawTools.heatmap(df)
    DrawTools.font_desc(ax=None, title=u'mtcars数据集的相关性矩阵', tick_rotation=45, tick_horizontalalignment='right')
    plt.show()


def test10():
    df = sns.load_dataset('iris')
    DrawTools.init()
    DrawTools.pairplot(df, 'species')
    plt.show()


test8()
