# -*- coding: utf-8 -*-
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

data = pd.read_csv('data.csv')
print data.info()
print data.head(5)
# axis 0删除行 1删除列
# inplace True 直接在原数据删除 False 生成一份新的删除后的数据
data.drop(["Cabin", "Name", "Ticket"], inplace=True, axis=1)
# 处理缺失值,使用随机森林，中值，平均值，
data["Age"] = data["Age"].fillna(data["Age"].mean())
# 只缺失1，2个值可以直接删除
data = data.dropna()
print data.info()
print data.head(5)
labels = data["Embarked"].unique().tolist()
data["Embarked"] = data["Embarked"].apply(lambda x: labels.index(x))
data["Sex"] = (data["Sex"] == "male").astype("int")
X = data.iloc[:, data.columns != "Survived"]
y = data["Survived"]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
for i in [x_train, x_test, y_train, y_test]:
    i.index = range(i.shape[0])



clf = tree.DecisionTreeClassifier(random_state=25)
clf.fit(x_train, y_train)

score = cross_val_score(clf, X, y, cv=10).mean()
print score
