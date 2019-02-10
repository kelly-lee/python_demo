# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np


def p(D, a, c=None):
    if c is None:
        return D[a].value_counts() / len(D[a])
    else:
        Size = D.groupby([c, a]).size().unstack()
        return Size / Size.sum()


def ent(P):
    return -P.apply(lambda p: p * np.log2(p)).sum()


def gain(D, a, c):
    return np.around(ent(p(D, c)) - p(D, a).mul(ent(p(D, a, c))).sum(), 3)


def iv(D, a):
    return ent(p(D, a))


def gain_ratio(D, a, c):
    IV = iv(D, a)
    if IV > 0:
        return gain(D, a, c) / IV
    else:
        return 0


def gini(P):
    return 1 - P.apply(lambda p: p ** 2).sum()


def gini_index(D, a, c):
    return np.around(p(D, a).mul(gini(p(D, a, c))).sum(), 3)


def arg_max_gain(D, A, c):
    df = pd.DataFrame()
    for a in A.keys():
        df = df.append(pd.DataFrame([[a, gain(D, a, c)]], columns=['a', 'gain']), ignore_index=True)
    df.sort_values(by=['gain'], ascending=False, inplace=True)
    return df.iloc[0]['a']


def arg_max_gain_ratio(D, A, c):
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
        self.av = av

    def child(self, child_node):
        self.child_nodes.append(child_node)

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
    a = arg_min_gini_index(D, A, c)
    node.branch(a)
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



def discretization(D, a):
    """
    离散化
    :param D:
    :param a:
    :return:
    """
    D_ = D.sort_values(['含糖率'])
    a4 = D_['含糖率']
    a1 = np.round((a4 + a4.shift(1)) / 2, 3)
    for a2 in a1:
        a3 = a4
        a3 = a3.where(a3 > a2, 0)
        a3 = a3.where(a3 <= a2, 1)
        D_['a1'] = a3
        print a2, gain(D_, 'a1', c)


if __name__ == '__main__':
    watermelone = pd.read_csv("watermelon_2.csv")
    D = watermelone
    columns = watermelone.columns.drop(['好瓜', '密度', '含糖率', '编号'])
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

    D_ = D.sort_values(['含糖率'])
    a4 = D_['含糖率']
    a1 = np.round((a4 + a4.shift(1)) / 2, 3)
    for a2 in a1:
        a3 = a4
        a3 = a3.where(a3 > a2, 0)
        a3 = a3.where(a3 <= a2, 1)
        D_['a1'] = a3
        print a2, gain(D_, 'a1', c)
    # pt = gain()
    # print np.around(ent(pt(D, c)) - pt(D, a1.index).mul(ent(pt(D, a1.index, c))).sum(), 3)

    # print type(p(D, c))

