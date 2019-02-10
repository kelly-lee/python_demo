# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np


def p(D, a, c=None):
    """
    样本数越多的分支结点影响越大

    :param D: 数据集
    :param a: 属性名
    :return: 权重 属性a各取值av的占比集
    """
    if c is None:
        return D[a].value_counts() / len(D[a])
    else:
        """
        色泽 乌黑  浅白  青绿
        好瓜
        否    2    4     3
        是    4    1     3
        """
        Size = D.groupby([c, a]).size().unstack()
        """
        色泽 乌黑  浅白  青绿
        好瓜
        否   2/6  4/5   3/6
        是   4/6  1/6   3/6
        """
        return Size / Size.sum()


def ent(P):
    """
    信息熵 Ent(D) = -Sum(Pklog2(Pk)),k=1~|y|
    |y| 分类集C【好瓜，坏瓜】的数量【2】

    :param P: 数据集D【西瓜集】在分类集C【好瓜，坏瓜】上的概率集【8/17,9/17】
    :return: 信息熵
    """
    return -P.apply(lambda p: p * np.log2(p)).sum()


def ent_da(D, a, c):
    """
    属性a中每个取值av的信息熵集合
    :param D: 数据集D【西瓜集】中所有在属性a【色泽】上取值为av【乌黑】的样本
    :param a: 属性名【色泽】
    :param c: 分类名【好瓜】
    :return: 属性a【色泽】中每个取值avs【青绿、乌黑、浅白】的信息熵集合【1.0000,0.918,0.722】
    """
    return ent(p(D, a, c))


def ent_d(D, c):
    """
    根结点的信息熵
    :param D: 数据集
    :param c: 分类名【好瓜】
    :return: 根结点的信息熵
    """
    return ent(p(D, c))


def gain(D, a, c):
    """
    信息增益 information gain
    ID3 (Iteratice Dichotomiser 迭代二分器) 决策树学习算法[Quinlan,1986]以信息增益为准则划分属性
    信息增益越大，使用a划分所获得的纯度提升越大
    Gain(D,a)= Ent(D)-Sum(|Dv|/|D|*Ent(Dv))
    Ent(Dv)：数据集D在属性a中每个取值av的信息熵集合
    |Dv|：数据集D中所有在属性a上取值为av的数量
    |D|：数据集D数量
    |Dv|/|D|：属性a每个取值av的权重
    Ent(D)：数据集D的信息熵
    属性a的信息增益
    :param D: 数据集
    :param a: 属性名【色泽】
    :param c: 分类名【好瓜】
    :return: 属性a【色泽】的信息增益gain【0.109】
    """
    return np.around(ent(p(D, c)) - p(D, a).mul(ent(p(D, a, c))).sum(), 3)


def arg_max_gain(D, A, c):
    """
    :param D: 数据集
    :param A: 属性名集
    :param c: 分类名
    :return: 属性集A中信息增益最大的属性a
    """
    df = pd.DataFrame()
    for a in A.keys():
        # 离散值处理
        if len(D[a]) == D[a].nunique():
            a_d_gain, a_def = discretization_gain(D, a, c)
            D[a] = D[a].where(D[a] > a_def, a_def)
            D[a] = D[a].where(D[a] <= a_def, 9999)
            A[a] = D[a].unique()
            df = df.append(
                pd.DataFrame([[a, a_d_gain]], columns=['a', 'gain']),
                ignore_index=True)
        else:
            df = df.append(pd.DataFrame([[a, gain(D, a, c)]], columns=['a', 'gain']), ignore_index=True)
    df.sort_values(by=['gain'], ascending=False, inplace=True)
    return df.iloc[0]['a']


def arg_max_gain_ratio(D, A, c):
    """
    :param D: 数据集
    :param A: 属性名集
    :param c: 分类名
    :return: 属性集A中信息增益大于等于平均信息增益，而增益率最大的属性a
    """
    df = pd.DataFrame()
    for a in A.keys():
        df = df.append(pd.DataFrame([[a, gain(D, a, c), gain_ratio(D, a, c)]], columns=['a', 'gain', 'gain_ratio']),
                       ignore_index=True)
    df.dropna(inplace=True)
    df = df[df['gain'] >= df['gain'].mean()].sort_values(by=['gain_ratio'], ascending=False)
    return df.iloc[0]['a']


def arg_min_gini_index(D, A, c):
    df = pd.DataFrame()
    for a in A.keys():
        df = df.append(pd.DataFrame([[a, gini_index(D, a, c)]], columns=['a', 'gini_index']), ignore_index=True)
    df.sort_values(by=['gini_index'], ascending=True, inplace=True)
    return df.iloc[0]['a']


def iv(D, a):
    """
    IV (Intrinsic Value 固有值 )[Quinlan,1993]
    :param D: 数据集
    :param a: 属性名
    :return:
    """
    return ent(p(D, a))


def gain_ratio(D, a, c):
    """
    :param D:
    :param a:
    :return:
    """
    IV = iv(D, a)
    return gain(D, a, c) / IV if IV > 0 else 0


def gini(P):
    return 1 - P.apply(lambda p: p ** 2).sum()


def gini_index(D, a, c):
    return np.around(p(D, a).mul(gini(p(D, a, c))).sum(), 3)


def discretization_gain(D, a, c):
    """
    离散化
    :param D:
    :param a:
    :return:
    """

    a_sort = D[a].sort_values()
    a_defs = np.round((a_sort + a_sort.shift(1)) / 2, 3)
    df = pd.DataFrame()
    for a_def in a_defs:
        a_discretization = a_sort.copy()
        a_discretization = a_discretization.where(a_discretization > a_def, -9999)
        a_discretization = a_discretization.where(a_discretization <= a_def, 9999)
        a_discretization = a_discretization.rename("a_d")
        Dv = D[[a, c]].join(a_discretization, how='outer')
        df = df.append(pd.DataFrame([[a_def, gain(Dv, 'a_d', c)]], columns=['a_def', 'a_d_gain']))
    df.dropna(inplace=True)
    df.sort_values(by=['a_d_gain'], ascending=False, inplace=True)
    return df.iloc[0]['a_d_gain'], df.iloc[0]['a_def']


class Node:
    def __init__(self):
        self.child_nodes = []
        self.a = 'root'
        self.av = 'root'

    def leaf(self, c):
        self.is_leaf = True
        self.c = c

    def branch(self, c):
        self.c = c

    def parent(self, a, av):
        self.a = a
        self.av = str(av)

    def child(self, child_node):
        self.child_nodes.append(child_node)

    def __str__(self):
        return self.c + ' ' + self.a + '=' + self.av

    def tostring(self, level=1):
        str = '[' + self.a + '=' + self.av + ']' + self.c + '\n'
        for child_node in self.child_nodes:
            for _ in range(level):
                str += '\t'
            str += child_node.tostring(level + 1)
        return str


def TreeGenerate(D, A, c):
    """
    :param D: 训练集
    :param A: 属性集
    :param c: 分类名
    :return: 决策树
    """
    # 生成结点node
    node = Node()
    # D中样本全属于同一类别C
    if D[c].nunique() == 1:
        # 将node标记为C类叶子结点
        cv = D[c].max()
        node.leaf(cv)
        return node
    # 如果A属性集为空，D中样本在A上取值相同
    if len(A) == 0:
        # 将node标记为叶子结点，其类别标记为D中样本数最多的类
        cv = D[c].max()
        node.leaf(cv)
        return node
    a = arg_max_gain(D, A, c)
    node.branch(a)
    print D
    # A的每个属性a取值必须是完整数据集的该属性a得所有取值
    for av in A[a]:
        Dv = D[D[a] == av]
        if len(Dv) == 0:
            child_node = Node()
            child_node.leaf(D[c].max())
        else:
            # 要拷贝，不同分支可能会有相同的子分支
            Av = A.copy()
            Av.pop(a)
            child_node = TreeGenerate(Dv, Av, c)
        child_node.parent(a, av)
        node.child(child_node)
    return node


if __name__ == '__main__':
    watermelone = pd.read_csv("watermelon_2.csv")
    D = watermelone
    columns = watermelone.columns.drop(['好瓜', '编号'])
    A = {}
    for column in columns.values:
        A[column] = D[column].unique()
    print A

    c = '好瓜'
    print '根结点的信息熵', np.round(ent_d(D, c), 3)
    for a in A.keys():
        print a + '的信息熵\n', np.round(ent_da(watermelone, a, c), 3)
    for a in A.keys():
        print a + '的信息增益', gain(D, a, c)
    a = arg_max_gain(D, A, c)
    print '信息增益最大属性', a
    Dv = D[D['纹理'] == '清晰']
    Av = A.copy()
    Av.pop('纹理')
    for av in Av.keys():
        print av + '的信息增益', gain(Dv, av, c)

    node = TreeGenerate(D, A, c)
    print node.tostring()

    for a in A.keys():
        print a + '增益率', iv(D, a)
    for a in A.keys():
        print a + '基尼指数', gini_index(D, a, c)

    # Dv = D[['含糖率', c]]
    # a_sort = D['含糖率'].sort_values()
    # a_defs = np.round((a_sort + a_sort.shift(1)) / 2, 3)
    # df = pd.DataFrame()
    # for a_def in a_defs:
    #     a_discretization = a_sort.copy()
    #     a_discretization = a_discretization.where(a_discretization > a_def, 0)
    #     a_discretization = a_discretization.where(a_discretization <= a_def, 1)
    #     Dv['含糖率_d'] = a_discretization
    #     df = df.append(pd.DataFrame([[a_def, gain(Dv, '含糖率_d', c)]], columns=['a_def', 'a_d']))
    # df.dropna(inplace=True)
    # df.sort_values(by=['a_d'], ascending=False, inplace=True)
    # print df.head(1)
    # print D
    # print Dv
    # pt = gain()
    # print np.around(ent(pt(D, c)) - pt(D, a1.index).mul(ent(pt(D, a1.index, c))).sum(), 3)
    print discretization_gain(D, '含糖率', c)
    print discretization_gain(D, '密度', c)

    # print type(p(D, c))
# for col_name in watermelone.columns:
#     if col_name == '好瓜':
#         continue
#     print col_name, gain(watermelone, '好瓜', col_name)
#
# print gain(watermelone[watermelone['纹理'] == '模糊'], '好瓜', '敲声')


# print ent_d(watermelone, '好瓜')
# print ent_dv(watermelone, '好瓜', '色泽')
# p = watermelone['色泽'].value_counts() / len(watermelone['色泽'])
# print np.around(-p.apply(lambda pk: pk * np.log2(pk)).sum(), 3)
#
# print watermelone.groupby('色泽')['好瓜'].value_counts().index
# print watermelone['色泽'].value_counts().index
# print -p.apply(lambda pk: pk * np.log2(pk)).sum()
