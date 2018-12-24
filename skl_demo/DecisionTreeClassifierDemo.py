# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
import sklearn.datasets as ds
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import graphviz

# 数据可视化，分类可视化，决策树可视化，参数对分类影响可视化

wine = ds.load_wine()
pd = pd.concat([pd.DataFrame(wine.data), pd.DataFrame(wine.target)], axis=1)
print pd

Xtrain, Xtest, Ytrain, Ytest = train_test_split(wine.data, wine.target, test_size=0.3)

# clf = tree.DecisionTreeClassifier(criterion="gini")

# clf.fit(Xtrain, Xtest)
# score = clf.score(Ytrain, Ytest)
# print score

# dot_data = tree.export_graphviz(clf
#                                 , filled=True
#                                 , rounded=True
#                                 , feature_names=['酒精', '苹果酸', '灰', '灰的碱性', '镁', '总酚', '类黄酮', '非黄烷类酚类', '花青素', '颜 色强度',
#                                                  '色调', 'od280/od315稀释葡萄酒', '脯氨酸']
#                                 , class_names=["琴酒", "雪莉", "贝尔摩德"]
#                                 # , feature_names=wine.feature_names
#                                 # , class_names=wine.target_names
#                                 )
# graph = graphviz.Source(dot_data)
# print graph

scores1 = []
scores2 = []
for i in range(100):
    clf = tree.DecisionTreeClassifier(max_depth=3
                                      , criterion="gini"
                                      , random_state=i
                                      , splitter="best"

                                      )
    clf.fit(Xtrain, Ytrain)
    score1 = clf.score(Xtrain, Ytrain)
    score2 = cross_val_score(clf, wine.data, wine.target, cv=10).mean()
    scores1.append(score1)
    scores2.append(score2)

plt.plot(range(1, 101), scores1, color="red", label="max_depth")
plt.plot(range(1, 101), scores2, color="blue", label="max_depth")
plt.legend()
plt.show()
