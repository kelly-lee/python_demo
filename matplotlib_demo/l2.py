#!/usr/bin/env python
# -*- coding: utf-8 -*-
# coding: utf-8
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

#https://www.jianshu.com/p/b02ec7dc39dd
def test1():
    np.random.seed(0)
    x1 = np.linspace(0, 10, 50)
    x2 = [0] * 10 + [1] * 40
    y = 2 * x1 + 5 + np.random.random(50) * 10

    print(matplotlib.matplotlib_fname())
    print(plt.rcParams['font.sans-serif'])
    # plt.rcParams['font.sans-serif'] = ['msyh']
    data = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})
    gridobj = sns.lmplot("x1", "y", data=data, hue="x2", legend=False)
    plt.legend(['类别0','类别1'])
    plt.show()


if __name__ == '__main__':
    test1()
    print(matplotlib.get_cachedir())