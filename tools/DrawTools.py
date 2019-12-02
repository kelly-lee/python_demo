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
from sklearn.metrics import roc_curve,auc

from tools import Preprocessing

#########
# 箱线图  x为分类标签(可以是文字或离散数字)，y为连续数值型特征(必须为数值，可离散)
# DrawTools.boxplot(data,feature_xs,feature_ys)
# 小提琴图 x为分类标签(可以是文字或离散数字)，y为连续数值型特征(必须为数值，可离散)
# DrawTools.violinplot(data, feature_xs, feature_ys)
# 线性拟合图
# DrawTools.regplot_grid(data, feature_xs, feature_y)
# 取值分布图
# DrawTools.barh_grid(data,  feature_xs,eature_h)

#########


large = 22;
med = 16;
small = 12;
font = FontProperties(fname='/System/Library/Fonts/PingFang.ttc', size=med)
params = {
    'figure.figsize': (12, 8),  # 图像的画布大小

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
def lmplot_grid(data, feature_xs=None, feature_y=None, grid=(4, 4)):
    flg = plt.figure()
    if feature_xs is None:
        feature_xs = Preprocessing.numeric_columns(data)
    for index, feature_x in enumerate(feature_xs):
        ax = flg.add_subplot(grid[0], grid[1], index + 1)
        lmplot(data, feature_x=feature_x, feature_y=feature_y, ax=ax)
    plt.show()


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
    print(data, feature_x, feature_y)
    gridobj = sns.regplot(feature_x, feature_y, data=data
                          # hue=feature_h
                          # , height=8  # 图像的高度（纵向，也叫做宽度）
                          # , aspect=1.6  # 图像的纵横比，因此 aspect*height = 每个图像的长度（横向），单位为英寸
                          # , legend=False
                          # , robust=True
                          # , col=feature_h
                          # , col_wrap=2
                          # , palette='tab10'  # 色板，tab10
                          # , scatter_kws=dict(s=60, linewidths=.7, edgecolors='black',
                          , ax=ax)

    gridobj.set(xlim=xlim, ylim=ylim)


def regplot(data, feature_x, feature_y, xlim=None, ylim=None, ax=None):
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
    gridobj = sns.regplot(feature_x, feature_y, data=data, ax=ax)

    # gridobj.set(xlim=xlim, ylim=ylim)


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
def violinplot(data, feature_x, feature_y, x_rotation=0, ax=None):
    if ax is None:
        ax = plt.gca()
    sns.violinplot(x=feature_x
                   , y=feature_y
                   , data=data
                   , ax=ax
                   , scale='width'
                   , inner='quartile')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=x_rotation)


# 箱线图
def boxplot(data, feature_x, feature_y=None, vertical=True, color=None, x_rotation=0, ax=None):
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
    ax.set_xticklabels(ax.get_xticklabels(), rotation=x_rotation)


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
    ax.set_xlim(-1, data[feature_x].quantile(q=0.999))
    ax.set_xlabel(feature_x)


# 密度图（多图）
def kdeplot_grid(data, feature_xs=None, feature_h=None, grid=None, col=5):
    """
    :param data:数据
    :param feature_xs:x轴特征
    :param feature_h:分组特征
    :param grid:网格分布，元组
    :return: 绘制密度图
    """
    if feature_xs is None:
        feature_xs = Preprocessing.numeric_columns(data)
        feature_xs = [feature for feature in feature_xs if data[feature].nunique() > 10]
    if grid is None:
        size = len(feature_xs)
        grid = (int(np.ceil(size / col)), col)

    flg = plt.figure()
    for index, feature_x in enumerate(feature_xs):
        ax = flg.add_subplot(grid[0], grid[1], index + 1)
        kdeplot(data, feature_x, feature_h, ax)
    plt.show()


# 直方密度图
def distplot(data, feature_x, feature_h=None, xlim=None, ylim=None, ax=None, bins=10, hist=True, kde=True,
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
def distplot_grid(data, feature_xs=None, feature_h=None, grid=(3, 3), bins=10, hist=True, kde=True):
    flg = plt.figure()
    if feature_xs is None:
        feature_xs = data.dtypes[(data.dtypes == 'int64') | (data.dtypes == 'float64')].index.tolist()
    for index, feature_x in enumerate(feature_xs):
        ax = flg.add_subplot(grid[0], grid[1], index + 1)
        distplot(data, feature_x=feature_x, feature_h=feature_h, bins=bins, hist=hist, kde=kde)
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
    for index, name in enumerate(feature_importance_names):
        print(name, sorted_feature_importances[index])
    # print([*zip(feature_importance_names, sorted_feature_importances)])

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


def hist(data, feature_x, feature_h, bins=None, xlim=None, ylim=None, ax=None):
    if ax is None:
        ax = plt.gca()
    if bins is None:
        labels = data[feature_x].unique().tolist()
        bins = data[feature_x].nunique()
    else:
        labels = None

    df_agg = data.loc[:, [feature_x, feature_h]].groupby(feature_h)
    vals = [df[feature_x].values.tolist() for i, df in df_agg]

    colors = [plt.cm.Spectral(i / float(len(vals) - 1)) for i in range(len(vals))]
    n, bs, patches = ax.hist(vals, bins, stacked=True, density=False, color=colors[:len(vals)])
    # Decoration
    ax.legend({group: col for group, col in zip(np.unique(data[feature_h]).tolist(), colors[:len(vals)])})
    ax.set_xlabel(feature_x)
    # plt.title(f"Stacked Histogram of ${x_var}$ colored by ${groupby_var}$", fontsize=22)
    ax.set_ylabel("Frequency")
    ax.set_xticks(ticks=bs)
    print(labels)
    if labels is not None:
        ax.set_xticklabels(labels=labels, rotation=45)
    else:
        ax.set_xticklabels(labels=bs, rotation=45)
    if xlim is not None:
        print(xlim)
        ax.set_xlim(0, 100)
    if ylim is not None:
        ax.set_ylim(ylim)
    # plt.ylim(0, 25)


def barh(data, feature_x, feature_h, sort=False, type='count', ax=None):
    Preprocessing.fillna(data, [feature_x], 'None')
    if ax is None:
        ax = plt.gca()
    size_data = data.groupby([feature_h, feature_x]).size().unstack()
    size_data.fillna(0, inplace=True)
    if type == 'p':
        size_data = size_data / size_data.sum()

    if sort:
        t = size_data.T
        if type == 'p':
            # 改为加权平均？？
            t['sort_val'] = 0.3 * t.iloc[:, -3] + 0.6 * t.iloc[:, -2] + 1 * t.iloc[:, -1]
        if type == 'count':
            t['sort_val'] = t.sum(axis=1)
        t = t.sort_values(by=['sort_val'])
        t.drop(columns=['sort_val'], inplace=True)
        size_data = t.T

    ticks = size_data.columns.astype(str).values  # 变成字符串才能正确显示排序
    labels = size_data.index.values
    n_label = len(labels)
    n_tick = len(ticks)
    colors = [plt.cm.Spectral(i / float(n_label - 1)) for i in range(n_label)]
    left = np.zeros(n_tick)
    for index in range(n_label):
        y = size_data.iloc[index, :].values
        ax.barh(ticks, y, height=0.9, left=left, color=colors[index], label=labels[index])
        ax.set_yticks(ticks=ticks)
        left += y
    ax.set_ylabel(feature_x)
    ax.set_xlabel('Percent')
    ax.legend()


def init_feature_cat(data, features=None, threshold=20):
    if features is None:
        feature_numeric = Preprocessing.numeric_columns(data)
        Preprocessing.fillna(data, feature_numeric, -1)
        feature_obj = Preprocessing.obj_columns(data)
        Preprocessing.fillna(data, feature_obj, 'None')
        features = feature_numeric + feature_obj
        features = [feature for feature in features if data[feature].nunique() <= threshold]
    return features


def init_feature_numeric(data, features=None, threshold=3):
    if features is None:
        features = Preprocessing.numeric_columns(data)
        features = [feature for feature in features if data[feature].nunique() >= threshold]
    return features


def init_grid(size, col, inches):
    col = np.min([size, col])
    grid = (int(np.ceil(size / col)), col)
    flg = plt.figure(figsize=(grid[1] * inches[0], grid[0] * inches[1]))
    return flg, grid


# 箱线图(网格)
def boxplot_grid(data, feature_xs=None, feature_ys=None, col=5, inches=(5, 4), x_rotation=0,
                 show_strip=False):
    feature_xs = init_feature_cat(data, feature_xs)
    feature_ys = init_feature_numeric(data, feature_ys)
    flg, grid = init_grid(len(feature_xs) * len(feature_ys), col, inches)

    index = 0
    for i, feature_x in enumerate(feature_xs):
        for j, feature_y in enumerate(feature_ys):
            ax = flg.add_subplot(grid[0], grid[1], index + 1)
            if (data[feature_y].dtypes == 'object'):
                data[feature_y] = data[feature_y].astype('category').cat.codes
            # print(feature_x, feature_y, index)
            boxplot(data, feature_x=feature_x, feature_y=feature_y, x_rotation=x_rotation, ax=ax)
            if show_strip:
                stripplot(data, feature_x=feature_x, feature_y=feature_y, ax=ax)
            index += 1
    plt.show()


# 小提琴图(网格)
def violinplot_grid(data, feature_xs=None, feature_ys=None, col=5, inches=(5, 4), x_rotation=0, show_strip=False):
    feature_xs = init_feature_cat(data, feature_xs)
    feature_ys = init_feature_numeric(data, feature_ys)
    flg, grid = init_grid(len(feature_xs) * len(feature_ys), col, inches)

    index = 0
    flg = plt.figure(figsize=(grid[1] * inches[0], grid[0] * inches[1]))
    for i, feature_x in enumerate(feature_xs):
        for j, feature_y in enumerate(feature_ys):
            ax = flg.add_subplot(grid[0], grid[1], index + 1)
            # print(feature_x, feature_y, index)
            if (data[feature_y].dtypes == 'object'):
                data[feature_y] = data[feature_y].astype('category').cat.codes
            violinplot(data, feature_x=feature_x, feature_y=feature_y, x_rotation=x_rotation, ax=ax)
            if show_strip:
                stripplot(data, feature_x=feature_x, feature_y=feature_y, ax=ax)
            index += 1
    plt.show()


def barh_grid(data, feature_xs=None, feature_h=None, col=5, inches=(6, 8), type='count', sort=True):
    feature_xs = init_feature_cat(data, feature_xs)
    flg, grid = init_grid(len(feature_xs), col, inches)

    for index, feature_x in enumerate(feature_xs):
        ax = flg.add_subplot(grid[0], grid[1], index + 1)
        barh(data, feature_x, feature_h, type=type, sort=sort, ax=ax)
    plt.show()


def regplot_grid(data, feature_xs=None, feature_y=None, col=5, inches=(5, 4)):
    feature_xs = init_feature_numeric(data, feature_xs, threshold=20)
    flg, grid = init_grid(len(feature_xs), col, inches)

    for index, feature_x in enumerate(feature_xs):
        print(feature_x, feature_y, index)
        ax = flg.add_subplot(grid[0], grid[1], index + 1)
        regplot(data, feature_x, feature_y, ax=ax)
    plt.show()


def xgb_eval_results(evals_result, col=2, inches=(5, 4)):
    flg, grid = None, None
    for i, label in enumerate(evals_result):
        eval_result = evals_result[label]
        if (i == 0):
            flg, grid = init_grid(len(eval_result), col, inches)
        for j, metric in enumerate(eval_result):
            ax = flg.add_subplot(grid[0], grid[1], j + 1)
            ax.plot(eval_result[metric], label=label)
            ax.set_ylabel(metric)
            ax.set_xlabel('n_tree')
            ax.grid(linewidth=0.5)
            ax.legend()
    plt.show()


def xgb_cv_results(cv_result, metrics, col=2, inches=(5, 4)):
    print(cv_result.columns)
    count = len(cv_result.columns)
    n_metrics = len(metrics)
    size = count / (n_metrics * 2)
    flg, grid = init_grid(size, col, inches)
    n_estimators = len(cv_result)
    x_axis = range(0, n_estimators)
    for i, metric in enumerate(metrics):
        ax = flg.add_subplot(grid[0], grid[1], i + 1)
        for group in ['train', 'test']:
            label = group + '-' + metric
            mean = cv_result.loc[:, group + '-' + metric + '-mean']
            std = cv_result.loc[:, group + '-' + metric + '-std']
            ax.errorbar(x_axis, mean, yerr=std, label=label)
        ax.grid(linewidth=0.5)
        ax.legend()
    plt.show()


def roc_auc_multiclass(y_true, y_pred_proba, n_class=None, ax=None):
    if ax is None:
        ax = plt.gca()
    if n_class is None:
        n_class = np.unique(y_true).size

    fprs_list = []
    tprs_list = []

    y_true = np.eye(n_class)[y_true.astype(int)]

    for i in range(n_class):
        print(y_true[:, i], y_pred_proba[:, i])
        fprs, tprs, thresholds = roc_curve(y_true[:, i], y_pred_proba[:, i])
        auc_ = auc(fprs, tprs)
        fprs_list.append(fprs)
        tprs_list.append(tprs)
        ax.plot(fprs, tprs, label='ROC curve of class{0}(area = {1:0.2f})'.format(i + 1, auc_))

    fpr_micro, tpr_micro, _ = roc_curve(y_true.ravel(), y_pred_proba.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)
    ax.plot(fpr_micro, tpr_micro, '--', lw=2,
            label='ROC curve of mirco(area = {1:0.2f})'.format(i + 1, auc_micro))

    fpr_macro = np.unique(np.concatenate([fprs_list[i] for i in range(n_class)]))
    mean_tpr = np.zeros_like(fpr_macro)
    for i in range(n_class):
        mean_tpr += np.interp(fpr_macro, fprs_list[i], tprs_list[i])
    tpr_macro = mean_tpr / n_class
    auc_macro = auc(fpr_macro, tpr_macro)
    ax.plot(fpr_macro, tpr_macro, '--', lw=2,
            label='ROC curve of macro(area = {1:0.2f})'.format(i + 1, auc_macro))
    ax.set_xlabel('False Positive Rate(fpr)')
    ax.set_ylabel('True Positive Rate(tpr)')
    ax.legend()
    plt.show()
