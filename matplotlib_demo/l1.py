#!/usr/bin/env python
# -*- coding: utf-8 -*-
# coding: utf-8
import sys

# reload(sys)
# sys.setdefaultencoding('utf8')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import train_test_split
from tools import DrawTools


# 画简单散点图
def test1():
    x = np.random.randn(10)
    y = x + x ** 2 - 10
    # 确定画布-当只有一个图的时候，不是必须存在的
    # figsize 画布大小
    plt.figure(figsize=(8, 4))
    plt.scatter(x, y, s=20,  # 点尺寸
                c='blue',  # 颜色
                label='Positive'  # 标签
                )
    plt.legend()  # 显示图例
    plt.show()  # 显示图形


# 散点图有不同颜色
def test2():
    x = np.random.rand(10, 2)
    y = np.array([0, 1, 1, 1, 3, 1, 0, 1, 4, 0])
    plt.figure(figsize=(8, 4))
    plt.scatter(x[:, 0], x[:, 1], s=50, c=y)
    plt.show()


# 散点图不同颜色有不同图例，要循环
def test3():
    x = np.random.rand(10, 2)
    y = np.array([0, 1, 2, 1, 3, 1, 0, 1, 4, 0])
    colors = ['red', 'yellow', 'blue', 'green', 'black']
    labels = ['A', 'B', 'C', 'D', 'E']
    for i in np.arange(5):
        plt.scatter(x[y == i, 0], x[y == i, 1], c=colors[i], label=labels[i])
    plt.legend()
    plt.show()


# 横纵坐标为特征，标签用颜色区分的散点图
def test4():
    # rural，urban 城市、乡村
    # ALH average，high，low
    large = 22;
    med = 16;
    small = 12;
    params = {'axes.titlesize': large,  # 子图上的标题字体大小
              'legend.fontsize': med,  # 图例的字体大小
              'figure.figsize': (16, 10),  # 图像的画布大小
              'axes.labelsize': med,  # 标签的字体大小
              'xtick.labelsize': med,  # x轴上的标尺的字体大小
              'ytick.labelsize': med,  # y轴上的标尺的字体大小
              'figure.titlesize': large}  # 整个画布的标题字体大小
    plt.rcParams.update(params)  # 设定各种各样的默认属性
    # 以下2行可以消除图例的边框
    plt.style.use('seaborn-whitegrid')  # 设定整体风格
    sns.set_style("white")  # 设定整体背景风格

    midwest = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/midwest_filter.csv")
    categorys = midwest['category']
    categorys_unique = categorys.unique()
    category_size = len(categorys_unique)
    colors = [plt.cm.tab10(i / float(category_size - 1)) for i in np.arange(category_size)]
    flg = plt.figure(figsize=(16, 10),
                     dpi=100,  # 图像分辨率
                     facecolor='w',  # 图像背景颜色，设置成白色，默认也是白色
                     edgecolor='w'  # 图像边框颜色，设置成黑色，默认也是黑色
                     )
    for index, category in enumerate(categorys_unique):
        label = str(category)
        data = midwest.loc[categorys == category, :]
        dx = midwest.loc[categorys == label, 'area']
        dy = midwest.loc[categorys == label, 'poptotal']
        s = 20,
        c = np.array(colors[index]).reshape(1, -1)
        # c = [[0.12156863 0.46666667 0.70588235 1.        ]] RGBA
        plt.scatter('area', 'poptotal', data=data, c=c, s=s, label=label)
    # plt.gca()获取当前子图，如果当前子图不存在，创建新子图
    plt.gca().set(xlim=(0.0, 0.12), ylim=(0, 80000))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Area', fontsize=22)
    plt.ylabel('Population', fontsize=22)
    plt.title('Scatterplot of Midwest Area vs Population', fontsize=22)
    plt.legend(fontsize=12)
    plt.show()


# 分析标签三个字母的含义，并用逻辑回归证明三个字母和哪些特征有关
def test5():
    midwest = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/midwest_filter.csv")
    midwest_1 = midwest.loc[midwest.category.isin(['AHR', 'HAU', 'LHU']), :]
    plt = DrawTools.drawA(midwest_1, feature_x='area', feature_y='poptotal', label_name='category',
                          feature_s='popasian', feature_text='county',
                          title='Scatterplot of Midwest Area vs Population')
    plt.show()

    midwest.columns = ["城市ID", "郡", "州", "面积", "总人口", "人口密度", "白人人口", "非裔人口", "美洲印第安人人口", "亚洲人口", "其他人种人口"
        , "白人所占比例", "非裔所占比例", "美洲印第安人所占比例", "亚洲人所占比例", "其他人种比例"
        , "成年人口", "具有高中文凭的比率", "大学文凭比例", "有工作的人群比例"
        , "已知贫困人口", "已知贫困人口的比例", "贫困线以下的人的比例", "贫困线以下的儿童所占比例", "贫困的成年人所占的比例", "贫困的老年人所占的比例"
        , "是否拥有地铁", "标签", "点的尺寸"]

    for i in range(3):
        midwest['c' + str(i)] = midwest['标签'].apply(lambda x: x[i])
    # 编码
    midwest.iloc[:, -3:] = OrdinalEncoder().fit_transform(midwest.iloc[:, -3:])
    midwest = midwest.loc[:, midwest.dtypes.values != 'O']  # O大写
    midwest.loc[:, midwest.dtypes.values == 'int64'] = midwest.loc[:, midwest.dtypes.values == 'int64'].astype(
        np.float64)
    midwest = midwest.iloc[:, [*range(1, 25), 26, 27, 28]]  # 删除'城市ID','点的尺寸'这一列
    # 标准化
    midwest.iloc[:, [*range(23)]] = StandardScaler().fit_transform(midwest.iloc[:, [*range(23)]])

    xtrain, xtest, ytrain, ytest = train_test_split(midwest.iloc[:, :-3], midwest.iloc[:, -3:], test_size=0.3,
                                                    random_state=420)
    for index in range(3):
        lr = LR(solver='newton-cg', multi_class='multinomial', random_state=420, max_iter=100 ** 20)
        lr = lr.fit(xtrain, ytrain.iloc[:, index])
        print(lr.score(xtrain, ytrain.iloc[:, index]))
        print(lr.score(xtest, ytest.iloc[:, index]))
        coef = pd.DataFrame(lr.coef_).T

        if index < 2:
            coef['mean'] = abs(coef).mean(axis=1)  # 为什么要abs？？？？
            coef['name'] = xtrain.columns
            coef.columns = ["Average", "High", "Low", "mean", "name"]
            coef = coef.sort_values(by='mean', ascending=False)

        else:
            coef.columns = ["value"]
            coef['name'] = xtrain.columns
            coef = coef.sort_values(by='value', ascending=False)
        print(coef.head())


# 简单凸包例子
def test6():
    # 产生正态分布的随机数
    np.random.seed(1)
    x1, y1 = np.random.normal(loc=5, scale=2, size=(2, 15))
    x2, y2 = np.random.normal(loc=8, scale=2.5, size=(2, 13))
    # 计算凸包
    # 画随机点
    plt.scatter(x1, y1)
    plt.scatter(x2, y2)
    DrawTools.drawPloygon(x1, y1, ax=None
                          , ec="k"
                          , fc="gold"
                          , alpha=0.1)
    DrawTools.drawPloygon(x2, y2, ax=None
                          , ec="lightblue"
                          , fc="none"
                          , linewidth=1.5)

    plt.show()




if __name__ == '__main__':
    np.random.seed(0)
    x1 = np.linspace(0, 10, 50)
    x2 = [0] * 10 + [1] * 40
    y = 2 * x1 + 5 + np.random.random(50) * 10
    import  matplotlib
    # print(matplotlib.matplotlib_fname())
    # print(plt.rcParams['font.sans-serif'])
    # plt.rcParams['font.sans-serif'] = ['Simhei']
    data = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})
    gridobj = sns.lmplot("x1", "y", data=data, hue="x2", legend=False)
    plt.legend(['类别0','类别1'])
    plt.show()
