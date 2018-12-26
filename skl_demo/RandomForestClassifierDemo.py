# -*- coding: utf-8 -*-
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 加载数据
cancer = datasets.load_breast_cancer()

# 构建随机森林分类期
clf = RandomForestClassifier()
scores = cross_val_score(clf, cancer.data, cancer.target, cv=10)
print scores
# 交叉验证，获得分数

# 调n_estimators，0～200 每10个调一下
# 画学习曲线
# scores_list = []
# for i in range(0, 200, 10):
#     print i
#     clf = RandomForestClassifier(n_estimators=i + 1
#                                  , random_state=90)
#     score_mean = cross_val_score(clf, cancer.data, cancer.target, cv=10).mean()
#     scores_list.append(score_mean)
#
# print (max(scores_list), scores_list.index(max(scores_list)) * 10 + 1)
# plt.plot(range(0, 200, 10), scores_list)
# plt.show()

clf = RandomForestClassifier(n_estimators=39
                             , random_state=90)
param_grids = [
    {'max_depth': np.arange(1, 20, 1)}
    , {'max_leaf_nodes': np.arange(25, 50, 1)}
    , {'criterion': ['gini', 'entropy']}
    , {'min_samples_split': np.arange(2, 2 + 20, 1)}
    , {'min_samples_leaf': np.arange(1, 1 + 10, 1)}
    , {'max_features': np.arange(5, 30, 1)}
]

for param_grid in param_grids:
    GS = GridSearchCV(clf, param_grid, cv=10)
    GS.fit(cancer.data, cancer.target)
    print GS.best_params_, GS.best_score_
