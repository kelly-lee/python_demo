#!/usr/bin/env python
# -*- coding: utf-8 -*-
# coding: utf-8
import sys

# from __future__ import division
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.datasets import load_breast_cancer, load_iris, load_boston
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from mlxtend.classifier import StackingClassifier
from sklearn import datasets, model_selection


def stacking_reg(clfs, X_train, X_test, y_train):
    n_folds = 10
    k_fold = KFold(n_splits=n_folds, random_state=420)
    n_model = len(clfs)
    blend_X_train = np.zeros((X_train.shape[0], n_model))  # (455, 5)
    blend_X_test = np.zeros((X_test.shape[0], n_model))  # (114, 5)
    for j, clf in enumerate(clfs):
        for i, (train_index, valid_index) in enumerate(k_fold.split(X_train)):
            X_train_kf = X_train[train_index]
            y_train_kf = y_train[train_index]
            clf.fit(X_train_kf, y_train_kf)
            X_valid_kf = X_train[valid_index]
            y_valid_kf = clf.predict(X_valid_kf)
            y_test = clf.predict(X_test)
            blend_X_train[valid_index, j] = y_valid_kf
            blend_X_test[:, j] += y_test
        blend_X_test[:, j] /= n_folds
    return blend_X_train, blend_X_test


# X_test=120,X_train=480
# blend_X_test = 90 ,3,blend_X_train = 360,3
def stacking_classes(clfs, n_classes, X_train, X_test, y_train):
    n_folds = 5
    k_fold = KFold(n_splits=n_folds, random_state=420)
    blend_X_train_list, blend_X_test_list = [], []
    for j, clf in enumerate(clfs):
        blend_X_train = np.zeros((X_train.shape[0], n_classes))  # (455, 5)
        blend_X_test = np.zeros((X_test.shape[0], n_classes))  # (114, 5)
        for i, (train_index, valid_index) in enumerate(k_fold.split(X_train)):
            X_train_kf = X_train[train_index]  # (108,4)
            y_train_kf = y_train[train_index]  # 108
            clf.fit(X_train_kf, y_train_kf)
            X_valid_kf = X_train[valid_index]  # (12,4)
            y_valid_kf = clf.predict_proba(X_valid_kf)  # (12,3)
            y_test = clf.predict_proba(X_test)  # 90
            blend_X_train[valid_index] = y_valid_kf  # (120,3)
            blend_X_test += y_test
            print(X_train_kf.shape, y_train_kf.shape, X_valid_kf.shape, y_valid_kf.shape)
        blend_X_test /= n_folds
        blend_X_train_list.append(blend_X_train.argmax(axis=1).reshape(-1, 1))
        blend_X_test_list.append(blend_X_test.argmax(axis=1).reshape(-1, 1))
    blend_X_train_list = np.hstack(blend_X_train_list)  # 120,15
    blend_X_test_list = np.hstack(blend_X_test_list)  # 120,15
    return blend_X_train_list, blend_X_test_list


def test_stacking_classes_multi():
    data = load_iris()
    X = data.data  # (569,30)
    y = data.target  # (569)
    shuffle = True

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0,
                                                        test_size=0.4)  # X_train.shape=(455, 30),X_test.shape=(114, 30)
    if shuffle:
        np.random.seed(0)  # seed to shuffle the train set
        idx = np.random.permutation(y_train.size)
        X_train = X_train[idx]
        y_train = y_train[idx]

    clfs = [RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
            RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
            GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=100)]
    blend_X_train, blend_X_test = stacking_classes(clfs, 3, X_train, X_test, y_train)

    print("Stacking.")
    clf = LogisticRegression()
    clf.fit(blend_X_train, y_train)
    print("Stacking Accuracy %0.6f:" % accuracy_score(y_test, clf.predict(blend_X_test)))
    from sklearn.metrics import classification_report

    classification_report(y_test, clf.predict(blend_X_test))
    n = 1
    for model in clfs:
        model.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)
        print("模型 %d Accuracy %0.6f:" % (n, accuracy_score(y_test, y_test_pred)))
        scores = model_selection.cross_val_score(clf, X, y,
                                                 cv=3, scoring='accuracy')
        print("Accuracy: %0.2f (+/- %0.2f) [%s]"
              % (scores.mean(), scores.std(), str(n)))
        # print(classification_report(y_test, y_test_pred))
        n = n + 1
    print(clf.coef_)


def test_stacking_classes():
    data = load_breast_cancer()
    X = data.data  # (569,30)
    y = data.target  # (569)
    shuffle = True

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0,
                                                        test_size=0.2)  # X_train.shape=(455, 30),X_test.shape=(114, 30)
    if shuffle:
        np.random.seed(0)  # seed to shuffle the train set
        idx = np.random.permutation(y_train.size)
        X_train = X_train[idx]
        y_train = y_train[idx]

    clfs = [RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
            RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
            GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=100)]
    blend_X_train, blend_X_test = stacking_classes(clfs, 2, X_train, X_test, y_train)

    print("Stacking.")
    clf = LogisticRegression(penalty='l2', solver='liblinear', max_iter=100)
    clf.fit(blend_X_train, y_train)
    print("Stacking Accuracy %0.6f:" % accuracy_score(y_train, clf.predict(blend_X_train)))
    print("Stacking Accuracy %0.6f:" % accuracy_score(y_test, clf.predict(blend_X_test)))
    n = 1
    for model in clfs:
        model.fit(X_train, y_train)
        print("模型 %d Accuracy %0.6f:" % (n, accuracy_score(y_test, model.predict(X_test))))
        n = n + 1
    print(clf.coef_)


def test_stacking_reg():
    data = load_boston()
    X = data.data  # (569,30)
    y = data.target  # (569)
    shuffle = True

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0,
                                                        test_size=0.2)  # X_train.shape=(455, 30),X_test.shape=(114, 30)
    if shuffle:
        np.random.seed(0)  # seed to shuffle the train set
        idx = np.random.permutation(y_train.size)
        X_train = X_train[idx]
        y_train = y_train[idx]

    clfs = [RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
            RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
            GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=100)]
    blend_X_train, blend_X_test = stacking_classes(clfs, 2, X_train, X_test, y_train)

    print("Stacking.")
    clf = LogisticRegression(penalty='l2', solver='liblinear', max_iter=100)
    clf.fit(blend_X_train, y_train)
    print("Stacking Accuracy %0.6f:" % accuracy_score(y_train, clf.predict(blend_X_train)))
    print("Stacking Accuracy %0.6f:" % accuracy_score(y_test, clf.predict(blend_X_test)))
    n = 1
    for model in clfs:
        model.fit(X_train, y_train)
        print("模型 %d Accuracy %0.6f:" % (n, accuracy_score(y_test, model.predict(X_test))))
        n = n + 1
    print(clf.coef_)


def test_mlxtend():
    data = load_iris()
    X = data.data  # (569,30)
    y = data.target  # (569)


    clfs = [RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
            RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
            GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=100)]
    lr = LogisticRegression()
    sclf = StackingClassifier(classifiers=clfs,
                              meta_classifier=lr)
    scores = model_selection.cross_val_score(sclf, X, y,
                                             cv=3, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]"
          % (scores.mean(), scores.std(), 'stacking'))

    print('3-fold cross validation:\n')

    for clf, label in zip(clfs,
                          ['KNN',
                           'Random Forest',
                           'Naive Bayes',
                           'StackingClassifier']):
        scores = model_selection.cross_val_score(clf, X, y,
                                                 cv=3, scoring='accuracy')
        print("Accuracy: %0.2f (+/- %0.2f) [%s]"
              % (scores.mean(), scores.std(), label))


if __name__ == '__main__':
    # data = load_breast_cancer()
    # X = data.data  # (569,30)
    # y = data.target  # (569)
    test_stacking_classes_multi()
    test_mlxtend()
