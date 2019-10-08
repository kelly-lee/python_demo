#!/usr/bin/env python
# -*- coding: utf-8 -*-
# coding: utf-8
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import patches
from scipy.spatial import ConvexHull
from matplotlib.font_manager import FontProperties
from scipy.stats import norm, skew

large = 22;
med = 16;
small = 12;
font = FontProperties(fname='/System/Library/Fonts/PingFang.ttc', size=med)
params = {
    'figure.figsize': (16, 12),  # 图像的画布大小

    'figure.titlesize': large,  # 整个画布的标题字体大小
    'axes.titlesize': large,  # 子图上的标题字体大小
    'legend.fontsize': med,  # 图例的字体大小
    'axes.labelsize': med,  # 标签的字体大小
    'xtick.labelsize': med,  # x轴上的标尺的字体大小
    'ytick.labelsize': med,  # y轴上的标尺的字体大小

    # 'font.family': ['sans-serif'],
    'font.sans-serif': ['Microsoft YaHei'],  # 字体
    'axes.unicode_minus': False,  # 显示负号

    'figure.facecolor': 'white',  # 前景色
    'figure.edgecolor': 'white',  # 边框色
    'figure.dpi': 100  # 分辨率
}


def init():
    plt.rcParams.update(params)  # 设定各种各样的默认属性
    # 以下2行可以消除图例的边框
    plt.style.use('seaborn-whitegrid')  # 设定整体风格
    sns.set_style("white", {'font.sans-serif': ['Microsoft YaHei']})  # 设定整体背景风格


# 点图
def scatter(data, feature_x, feature_y, feature_c=None, feature_s=None, ax=None, xlim=None, ylim=None):
    """
    :param data: 数据集
    :param feature_x: x轴特征
    :param feature_y: y轴特征名
    :param feature_c: 点颜色特征名
    :param feature_s: 点大小特征名
    :param feature_text: 点文字特征名
    :param title: 图片名称
    :return: 横纵坐标为特征，标签用颜色区分的散点图
    """

    if feature_s is None:
        feature_s = 20
    if ax is None:
        # plt.gca()获取当前子图，如果当前子图不存在，创建新子图
        ax = plt.gca()
    if xlim is None:
        xlim = (data[feature_x].min(), data[feature_x].max())
    if ylim is None:
        ylim = (data[feature_y].min(), data[feature_y].max())

    if feature_c is not None:
        labels = data[feature_c]
        label_unique = labels.unique()
        label_unique_size = len(label_unique)
        colors = [plt.cm.tab10(i / float(label_unique_size - 1)) for i in np.arange(label_unique_size)]
        for index, label in enumerate(label_unique):
            d = data.loc[labels == label, :]
            c = np.array(colors[index]).reshape(1, -1)
            # c = [[0.12156863 0.46666667 0.70588235 1.        ]] RGBA
            ax.scatter(feature_x,
                       feature_y,
                       data=d,
                       c=c,
                       # edgecolors='black',  # 点边缘的颜色
                       linewidth=1,
                       s=feature_s,  # s里面可以输入和坐标轴长度不一致的序列
                       # 如果输入了比原始数据更长的序列，参数只会截取和横纵坐标一样长的对应尺寸
                       alpha=0.5,  # 透明度
                       label=str(label))
    else:
        ax.scatter(feature_x,
                   feature_y,
                   data=data,
                   # edgecolors='black',  # 点边缘的颜色
                   linewidth=1,
                   s=feature_s,  # s里面可以输入和坐标轴长度不一致的序列
                   # 如果输入了比原始数据更长的序列，参数只会截取和横纵坐标一样长的对应尺寸
                   alpha=1  # 透明度
                   )
    ax.set(xlim=xlim, ylim=ylim)
    # ax.set_xticklabels(labels=[], fontdict={'fontsize': small})
    # ax.set_yticklabels(labels=[], fontdict={'fontsize': small})


# 文字描述
def font_desc(ax=None, title='', xlabel='', ylabel='', legends=None, title_fontsize=large, label_fontsize=med,
              tick_fontsize=small, tick_rotation=0, tick_horizontalalignment='center'):
    if ax is None:
        ax = plt.gca()
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.set_title(title, fontsize=title_fontsize)

    # ax.title.set_fontsize(title_fontsize)
    # ax.xaxis.label.set_fontsize(label_fontsize)
    # ax.yaxis.label.set_fontsize(label_fontsize)

    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles, legends)

    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_horizontalalignment(tick_horizontalalignment)
    ax.tick_params(axis='x', labelsize=tick_fontsize, rotation=tick_rotation)
    ax.tick_params(axis='y', labelsize=tick_fontsize, rotation=tick_rotation)


# 写字
def text(data, feature_x, feature_y, feature_text, ax, **kwargs):
    if ax is None:
        ax = plt.gca()
    for index, row in data.iterrows():
        ax.text(row[feature_x], row[feature_y], s=row[feature_text], fontdict={'fontsize': small},
                horizontalalignment='center', **kwargs)


# 凸包
def ploygon(data, feature_x, feature_y, ax, **kwargs):
    '''
    :param data:数据集
    :param feature_x:x轴特征名
    :param feature_y:x轴特征名
    :param ax:
    :param kwargs:
    :return: 绘制凸包
    '''
    if ax is None:
        ax = plt.gca()
    p = np.c_[data[feature_x].values, data[feature_y].values]
    hull = ConvexHull(p)
    poly = plt.Polygon(p[hull.vertices, :], **kwargs)
    ax.add_patch(poly)


# 线性拟合图（多图）
def lmplot_mul(data, feature_xs=None, feature_y=None, grid=(4, 4)):
    flg = plt.figure()
    if feature_xs is None:
        feature_xs = data.dtypes[(data.dtypes == 'int64') | (data.dtypes == 'float64')].index.tolist()
    for index, feature_x in enumerate(feature_xs):
        ax = flg.add_subplot(grid[0], grid[1], index + 1)
        lmplot(data, feature_x=feature_x, feature_y=feature_y, ax=ax)


# 线性拟合图
def lmplot(data, feature_x, feature_y, feature_h=None, xlim=None, ylim=None, ax=None):
    '''
    :param data: 数据集
    :param feature_x: x轴特征名
    :param feature_y: y轴特征名
    :param feature_h: 分组特征名
    :param xlim: x轴取值范围（二元组）
    :param yDrawTools.lmplot(train_data,'OverallQual','SalePrice')lim: y轴取值范围（二元组）
    :return: 绘制线性拟合图
    '''
    if ax is None:
        ax = plt.gca()
    if xlim is None:
        xlim = (data[feature_x].min(), data[feature_x].max())
    if ylim is None:
        ylim = (data[feature_y].min(), data[feature_y].max())
    gridobj = sns.lmplot(feature_x, feature_y, data=data, hue=feature_h
                         , height=8  # 图像的高度（纵向，也叫做宽度）
                         , aspect=1.6  # 图像的纵横比，因此 aspect*height = 每个图像的长度（横向），单位为英寸
                         , legend=False
                         # , robust=True
                         # , col=feature_h
                         # , col_wrap=2
                         , palette='tab10'  # 色板，tab10
                         , scatter_kws=dict(s=60, linewidths=.7, edgecolors='black')
                         )
    gridobj.set(xlim=xlim, ylim=ylim)


# 计数图
def stripplot(data, feature_x, feature_y, ax=None):
    """
    :param data: 数据集
    :param feature_x:  x轴特征名
    :param feature_y: y轴特征名
    :param ax:
    :return: 绘制计数图
    """
    if ax is None:
        ax = plt.gca()
    # data_group = data.groupby([feature_x, feature_y]).size().reset_index(name='counts')
    sns.stripplot(x=feature_x,
                  y=feature_y
                  # , size=data_group['counts']
                  , data=data
                  , ax=ax
                  , linewidth=1.5
                  # , palette='tab10'
                  , jitter=1
                  , color='black'
                  )


# 小提琴图
def violinplot(data, feature_x, feature_y, ax=None):
    if ax is None:
        ax = plt.gca()
    sns.violinplot(x=feature_x
                   , y=feature_y
                   , data=data
                   , ax=ax
                   , scale='width'
                   , inner='quartile')


# 小提琴图(多图)
def violinplot_mul(data, feature_x, feature_ys=None, grid=(2, 2), show_strip=True):
    flg = plt.figure()
    if feature_ys is None:
        feature_ys = data.dtypes[(data.dtypes == 'int64') | (data.dtypes == 'float64')].index.tolist()
    for index, feature_y in enumerate(feature_ys):
        ax = flg.add_subplot(grid[0], grid[1], index + 1)
        violinplot(data, feature_x=feature_x, feature_y=feature_y, ax=ax)
        if show_strip:
            stripplot(data, feature_x=feature_x, feature_y=feature_y, ax=ax)


# 箱线图
def boxplot(data, feature_x, feature_y=None, vertical=True, color=None, ax=None):
    """
    :param data: 数据集
    :param feature_x: x轴特征名
    :param vertical: 图形走向是否垂直， 默认水平
    :param color: 箱体颜色
    :param ax:
    :return: 绘制箱线图
    """
    if ax is None:
        ax = plt.gca()
    orient = 'v' if vertical else 'h'
    sns.boxplot(data=data, x=feature_x, y=feature_y, ax=ax, orient=orient, color=color)


# 箱线图(多图)
def boxplot_mul(data, feature_xs, feature_y=None, grid=(2, 2), show_strip=True):
    flg = plt.figure()
    if feature_xs is None:
        feature_xs = data.dtypes[(data.dtypes == 'int64') | (data.dtypes == 'float64')].index.tolist()
    for index, feature_x in enumerate(feature_xs):
        ax = flg.add_subplot(grid[0], grid[1], index + 1)
        boxplot(data, feature_x=feature_x, feature_y=feature_y, ax=ax)
        if show_strip:
            stripplot(data, feature_x=feature_x, feature_y=feature_y, ax=ax)


def hist(data, feature_x, bins=10, orientation='horizontal', color='black', invert=False, ax=None):
    """
    :param data:  数据集
    :param feature_x:  x轴特征名
    :param bins: 分箱数
    :param orientation: 图形走向 (horizontal|vertical)
    :param color: 分箱颜色
    :param ax:
    :return: 绘制分箱图
    """
    if ax is None:
        ax = plt.gca()
    ax.hist(data[feature_x], bins, histtype='stepfilled', orientation=orientation, color=color)
    if invert:
        ax.invert_yaxis()


# 分箱图
def hist(data, feature_x, bins=10, vertical=False, color='black', invert_x=False, invert_y=False, ax=None):
    """
    :param data:  数据集
    :param feature_x:  x轴特征名
    :param bins: 分箱数
    :param vertical: 图形走向是否垂直， 默认水平
    :param color: 分箱颜色
    :param invert_x: 是否翻转x轴
    :param invert_y: 是否翻转y轴
    :param ax:
    :return: 绘制分箱图
    """
    if ax is None:
        ax = plt.gca()
    orientation = 'vertical' if vertical else 'horizontal'

    ax.hist(data[feature_x], bins, histtype='stepfilled', orientation=orientation, color=color)
    if invert_y:
        ax.invert_yaxis()
    if invert_x:
        ax.invert_xaxis()


# 热力图
def heatmap(data, ax=None, cmap='RdYlGn'):
    if ax is None:
        ax = plt.gca()
    coef = data.corr()
    sns.heatmap(data=coef,  # 相关性矩阵
                cmap=cmap,  # 颜色
                xticklabels=coef.columns,  # 横轴文字
                yticklabels=coef.columns,  # 纵轴文字
                annot=True,  # 文字，True表示显示相关性系数，也可以是数值矩阵
                center=0,  # 热力条的中间值
                vmin=-1,  # 热力条最小值 类似xlim.min
                vmax=1,  # 热力条最大值 类似xlim.max
                ax=ax
                )

    # plt.imshow(coef, interpolation='nearest', cmap=cmap
    #            )
    # plt.colorbar()


# 矩阵图
def pairplot(data, feature_h, ax=None):
    sns.pairplot(data  # 数据，各个特征和标签
                 , kind="scatter"  # 要绘制的图像类型
                 , hue=feature_h  # 类别所在的列（标签）
                 , plot_kws=dict(s=40, edgecolor="white", linewidth=1)  # 散点图的一些详细设置
                 )


# 密度图
def kdeplot(data, feature_x, feature_h=None, ax=None):
    """
    :param data:
    :param feature_x:
    :param feature_h:
    :param ax:
    :return: 绘制密度图
    """
    if ax is None:
        ax = plt.gca()
    if feature_h is None:
        sns.kdeplot(data[feature_x]
                    , shade=True  # 填空阴影
                    # , vertical=True
                    , color='black'
                    , alpha=0.5
                    , ax=ax
                    )
    else:
        hues = data[feature_h].unique()
        hue_size = len(hues)
        for index, hue in enumerate(hues):
            x = data.loc[data[feature_h] == hue, feature_x]
            sns.kdeplot(x
                        , shade=True  # 填空阴影
                        # , vertical=True
                        , color=plt.cm.Paired(index / float(hue_size - 1))
                        , alpha=0.8
                        , label=hue
                        # , linewidth=3
                        # , linestyle='--'
                        , ax=ax
                        )


# 密度图（多图）
def kdeplot_mul(data, feature_xs=None, feature_h=None, grid=(3, 3)):
    """
    :param data:数据
    :param feature_xs:x轴特征
    :param feature_h:分组特征
    :param grid:网格分布，元组
    :return: 绘制密度图
    """
    if feature_xs is None:
        feature_xs = data.dtypes[(data.dtypes == 'int64') | (data.dtypes == 'float64')].index.tolist()
    flg = plt.figure()
    for index, feature_x in enumerate(feature_xs):
        ax = flg.add_subplot(grid[0], grid[1], index + 1)
        kdeplot(data, feature_x, feature_h, ax)
        ax.set_xlabel(feature_x)


# 直方密度图
def displot(data, feature_x, feature_h=None, xlim=None, ylim=None, ax=None, bins=10, hist=True, kde=True,
            color='black'):
    if ax is None:
        ax = plt.gca()
    if feature_h is None:
        sns.distplot(data[feature_x]
                     , bins=bins
                     , color=color
                     , hist=hist
                     , kde=kde
                     , fit=norm
                     # , hist_kws={'color': 'g', 'histtype': 'bar', 'alpha': 0.4}
                     # , kde_kws={'color': 'r', 'linestyle': '-.', 'linewidth': 3, 'alpha': 0.7}
                     )
    else:
        hues = data[feature_h].unique()
        hue_size = len(hues)
        for index, hue in enumerate(data[feature_h].unique()):  # 防止hue是文字，颜色就用index获得
            x = data.loc[data[feature_h] == hue, feature_x]
            sns.distplot(x
                         , bins=bins
                         , color=plt.cm.Paired(index / float(hue_size - 1))
                         , hist=hist
                         , kde=kde
                         , label=hue
                         # , hist_kws={'color': 'g', 'histtype': 'bar', 'alpha': 0.4}
                         , kde_kws={'linewidth': 3, 'alpha': 1}
                         )
    if xlim is not None:
        ax.set(xlim=xlim)
    if ylim is not None:
        ax.set(ylim=ylim)


# 直方密度图（多图）
def displot_mul(data, feature_xs=None, feature_h=None, grid=(3, 3), bins=10, hist=True, kde=True):
    flg = plt.figure()
    if feature_xs is None:
        feature_xs = data.dtypes[(data.dtypes == 'int64') | (data.dtypes == 'float64')].index.tolist()
    for index, feature_x in enumerate(feature_xs):
        ax = flg.add_subplot(grid[0], grid[1], index + 1)
        displot(data, feature_x=feature_x, feature_h=feature_h, bins=bins, hist=hist, kde=kde)
        ax.set_xlabel(feature_x)
        ax.legend()


# def draw_miss_rate():


def hlines(data, feature_column, val_column, ax=None):
    if ax is None:
        ax = plt.gca()
    ax.hlines(y=data.index, xmin=0, xmax=data[val_column], color='firebrick', alpha=0.4, linewidth=30)
    for index, row in data.iterrows():
        plt.text(row[val_column], index, round(row[val_column], 2), horizontalalignment='left',
                 verticalalignment='center', fontdict={'color': 'black', 'fontsize': 30})
    plt.yticks(data.index, data[feature_column], fontsize=30)
    plt.show()


def feature_importance(feature_importances, columns):
    sorted_feature_importances = feature_importances[np.argsort(-feature_importances)]
    feature_importance_names = columns[np.argsort(-feature_importances)]
    print([*zip(feature_importance_names, sorted_feature_importances)])

    fi = pd.DataFrame({'name': feature_importance_names, 'score': sorted_feature_importances})
    # fi = pd.DataFrame([*zip(feature_importance_names, sorted_feature_importances)], columns=['name', 'score'])
    fi = fi.sort_values(by=['score'], ascending=True)
    fi = fi.reset_index(drop=True)
    fi = fi.tail(30)

    hlines(fi, 'name', 'score')
    # ax = plt.gca()
    # ax.hlines(y=fi.index, xmin=0, xmax=fi.score, color='firebrick', alpha=0.4, linewidth=30)
    # for index, row in fi.iterrows():
    #     plt.text(row['score'], index, round(row['score'], 2), horizontalalignment='left',
    #              verticalalignment='center', fontdict={'color': 'black', 'fontsize': 30})
    #
    # plt.yticks(fi.index, fi.name, fontsize=30)
    # # ax.scatter(x=fi.index, y=fi.score, s=75, color='firebrick', alpha=0.7)
    # plt.show()
