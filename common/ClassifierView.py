# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import decomposition
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import math
from sklearn.datasets import make_moons
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html#sphx-glr-auto-examples-datasets-plot-iris-dataset-py
def viewCalssifierData(data, target, axis):
    data_reduced = data
    # data_reduced = decomposition.PCA(n_components=2).fit_transform(data)
    x = data_reduced[:, 0]
    y = data_reduced[:, 1]
    x_min, x_max = x.min() - 0.5, x.max() + 0.5
    y_min, y_max = y.min() - 0.5, y.max() + 0.5
    axis.scatter(x, y, c=target, edgecolors='k')
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(y_min, y_max)
    axis.set_xticks(())
    axis.set_yticks(())


#
def viewCalssifier(clf, data, target, axis):
    # 特征集降成2维
    data_reduced = decomposition.PCA(n_components=2).fit_transform(data)
    # 特征集x
    datax = data_reduced[:, 0]
    # 特征集y
    datay = data_reduced[:, 1]
    datax_min, datax_max = datax.min() - 0.5, datax.max() + 0.5
    datay_min, datay_max = datay.min() - 0.5, datay.max() + 0.5


    data_train, data_test, target_train, target_test = train_test_split(data_reduced, target, test_size=0.3)
    clf.fit(data_train, target_train)
    score = clf.score(data_test, target_test)
    h = 0.02
    gridx, gridy = np.meshgrid(np.arange(datax_min, datax_max, h), np.arange(datay_min, datay_max, h))
    z = clf.predict(np.c_[gridx.ravel(), gridy.ravel()]).reshape(gridx.shape)


    axis.contourf(gridx, gridy, z, alpha=0.8)
    axis.scatter(datax, datay, c=target, edgecolors='k')
    axis.text(datax_max - .3, datay_min + .3, ('%.2f' % score).lstrip('0'),
            size=15, horizontalalignment='right')

    axis.set_xlim(datax_min, datax_max)
    axis.set_ylim(datay_min, datay_max)
    axis.set_xticks(())
    axis.set_yticks(())


def viewCalssifierData3D(data, target):
    fig = plt.figure(figsize=(4, 3))
    ax = Axes3D(fig, elev=-150, azim=110)
    data_reduced = decomposition.PCA(n_components=3).fit_transform(data)
    ax.scatter(data_reduced[:, 0], data_reduced[:, 1], c=target, edgecolors='k')


# moons = datasets.make_moons(noise=0.3, random_state=0)
# circles = datasets.make_circles(noise=0.2, factor=0.5, random_state=1)
# datasets = [moons, circles]
# knc = KNeighborsClassifier(3)
# dtc = DecisionTreeClassifier(max_depth=5)
# rfc = RandomForestClassifier()
# classifiers = [knc, dtc, rfc]
#
# row = len(datasets)
# col = len(classifiers) + 1

# plt.figure(figsize=(4 * row, 2 * col))
# for i in range(row):
#     for j in range(col):
#         print i, j
#         ax = plt.subplot(row, col, i * col + j + 1)
#         x, y = datasets[i]
#         if j == 0:
#             viewCalssifierData(x, y, ax)
#         else:
#             viewCalssifier(classifiers[j - 1], x, y, ax)
#         print i, j
# plt.show()
# # viewData3D(iris.data[:, :3], iris.target)
# # viewData(iris.data[:, :2], iris.target)
# # plt.show()
# #
#
# # view1(wine.data[:, 0], wine.data[:, 1], wine.target)
#
# # iris = datasets.load_iris()
# # wine = datasets.load_wine()
# moons = datasets.make_moons(noise=0.3, random_state=0)
# circles = datasets.make_circles(noise=0.2, factor=0.5, random_state=1)
# datasets = [moons, circles]
# knc = KNeighborsClassifier(3)
# dtc = tree.DecisionTreeClassifier(max_depth=5)
# classifiers = [knc, dtc]
#
# row = len(datasets)
# col = len(classifiers) + 1
#
# # print range(row)
# plt.figure(figsize=(4 * row, 2 * col))
# for i in range(row):
#     for j in range(col):
#         print i, j
#         ax = plt.subplot(row, col, i * col + j + 1)
#         x, y = datasets[i]
#         if j == 0:
#             viewCalssifierData(x, y, ax)
#         else:
#             viewCalssifier(classifiers[j - 1], x, y, ax)
#         print i, j
#
# # x, y = moons
# #
# # clf = tree.DecisionTreeClassifier(max_depth=5)
# # viewCalssifier(clf, x, y, ax1)
# #
# # ax2 = plt.subplot(1, 2, 2)
# #
# # x, y = circles
# # clf = tree.DecisionTreeClassifier(max_depth=5)
# # viewCalssifier(clf, x, y, ax2)
# plt.show()
