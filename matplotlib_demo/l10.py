#!/usr/bin/env python
# -*- coding: utf-8 -*-
# coding: utf-8
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tools import DrawTools


def test1():
    n = np.random.rand(100)
    sns.kdeplot(n
                , shade=True  # 填空阴影
                # , vertical=True
                , color='red'
                , alpha=0.8
                , linewidth=3
                , linestyle='--'
                )
    plt.hist(n)
    plt.show()


def test2():
    df = pd.read_csv("https://github.com/selva86/datasets/raw/master/mpg_ggplot2.csv")
    name = ["汽车制造商", "型号名称", "发动机排量(L)", "制造年份", "气缸数量", "手动/自动"
        , "驱动类型", "城市里程/加仑", "公路里程/加仑", "汽油种类", "车辆类型"]
    print([*zip(df.columns, name)])
    DrawTools.kdeplot(df, 'cty', 'cyl')
    plt.show()


def test3():
    df = pd.read_csv("https://github.com/selva86/datasets/raw/master/mpg_ggplot2.csv")
    DrawTools.kdeplot_mul(df, ['cty', 'displ', 'hwy'], 'cyl', (2, 2))
    plt.show()


def test4():
    df = pd.read_csv("https://github.com/selva86/datasets/raw/master/mpg_ggplot2.csv")
    DrawTools.init()
    DrawTools.displot(df, 'cty', 'class', xlim=(5, 35), ylim=(0, 0.8))
    plt.legend()
    plt.show()


def test5():
    df = pd.read_csv("https://github.com/selva86/datasets/raw/master/mpg_ggplot2.csv")
    DrawTools.init()
    DrawTools.stripplot(df, 'class', 'hwy')
    DrawTools.boxplot(df, 'class', 'hwy', vertical=True)

    # sns.boxplot(x='class', y='hwy', data=df, notch=False)
    plt.show()


def test5():
    df = pd.read_csv("https://github.com/selva86/datasets/raw/master/mpg_ggplot2.csv")
    DrawTools.init()
    DrawTools.stripplot(df, 'class', 'hwy')
    DrawTools.boxplot(df, 'class', 'hwy', vertical=True)
    plt.show()

def test6():
    df = pd.read_csv("https://github.com/selva86/datasets/raw/master/mpg_ggplot2.csv")
    DrawTools.init()
    DrawTools.violinplot(df, 'class', 'hwy')
    DrawTools.stripplot(df, 'class', 'hwy')
    plt.show()

test5()
