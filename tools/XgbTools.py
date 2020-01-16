#!/usr/bin/env python
# -*- coding: utf-8 -*-
# coding: utf-8
import sys

import numpy as np
import xgboost as xgb
from sklearn import datasets
from sklearn.metrics import log_loss, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from xgboost import XGBClassifier
import matplotlib.pyplot  as plt
import sklearn.metrics
from tools import ModelTools, DrawTools


def xgb_cv(X, y, params, num_boost_round=300, metrics=None, nfold=5, stratified=True, shuffle=True, fpreproc=None,
           missing=None, weight=None):
    d_matrix = xgb.DMatrix(X, label=y, missing=missing, weight=weight)
    cv_result = xgb.cv(params,
                       d_matrix,
                       num_boost_round=num_boost_round,
                       nfold=nfold, stratified=stratified, shuffle=shuffle,
                       metrics=metrics,
                       early_stopping_rounds=np.ceil(num_boost_round / 10),
                       fpreproc=fpreproc,
                       verbose_eval=10)
    return cv_result


def xgbc_grid_search_cv(X, y, objective=None, num_class=None, init_param=None, grid_params=None,
                        metrics=['merror', 'mlogloss'],
                        scoring='neg_log_loss',
                        shuffle=True, random_state=0,
                        fpreproc=None, weight=None):
    if init_param is None:
        init_param = {
            'objective': objective,
            'num_class': num_class,
        }
    if grid_params is None:
        grid_params = [
            # {'n_estimators': []},
            {'max_depth': np.arange(2, 10, 2),
             'min_child_weight': np.arange(1, 6, 2)},
            {'n_estimators': [],
             'learning_rate': np.arange(0.01, 0.1, 0.02)},
            {'subsample': np.arange(0.5, 1.01, 0.1),
             'colsample_bytree': np.arange(0.5, 1.01, 0.1)},
            {'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100],  # 越大越好
             'reg_lambda': [1e-5, 1e-2, 0.1, 1, 100]}  # 越小越好
        ]
    model = XGBClassifier(**init_param)
    kflod = StratifiedKFold(n_splits=5, shuffle=False, random_state=0)

    for grid_param in grid_params:
        grid_search_cv(X, y, model, grid_param, metrics=metrics, scoring=scoring, kflod=kflod, fpreproc=fpreproc,
                       weight=weight)
    return model


# higgsboson
def grid_search_cv(X, y, model, grid_param, metrics, scoring, kflod, fpreproc=None, weight=None):
    if (grid_param.get('n_estimators') is not None):
        cv_result = xgb_cv(X, y, model.get_xgb_params(), num_boost_round=model.n_estimators, metrics=metrics,
                           fpreproc=fpreproc,
                           missing=model.missing,
                           weight=weight)
        DrawTools.xgb_cv_results(cv_result, metrics)
        model.n_estimators = len(cv_result)
        grid_param.pop('n_estimators')
        print('best:', {'n_estimators': len(cv_result)},
              cv_result.loc[len(cv_result) - 1, 'test-' + metrics[0] + '-mean'])
    if len(grid_param) > 0:
        print(grid_param)
        gsearch = GridSearchCV(model, param_grid=grid_param, scoring=scoring, cv=kflod, return_train_score=True)
        gsearch.fit(X, y)
        cv_results = gsearch.cv_results_
        mean_test_score = cv_results['mean_test_score']
        mean_train_score = cv_results['mean_train_score']
        plt.xticks(range(0, len(cv_results['params'])),
                   labels=[[round(param, 3) for param in params.values()] for params in cv_results['params']],
                   rotation=90)
        plt.plot(mean_test_score, label='test')
        plt.plot(mean_train_score, label='train')
        plt.ylabel(scoring)
        plt.xlabel(list(cv_results['params'][0].keys()))
        plt.legend()
        plt.grid()
        plt.show()

        # param_grid_max_depth = param_grid_1['max_depth']
        # param_grid_min_child_weight = param_grid_1['min_child_weight']

        # mean_test_score = cv_results['mean_test_score'].reshape(len(param_grid_max_depth), len(param_grid_min_child_weight))
        # for i, value in enumerate(param_grid_max_depth):
        #     print(-mean_test_score[i])
        #     plt.plot(param_grid_min_child_weight, mean_test_score[i], label=value)
        # plt.xlabel('min_child_weight')
        # plt.legend()
        # plt.show()

        print('best:', gsearch.best_params_, gsearch.best_score_)
        model.set_params(**gsearch.best_params_)
    return model


def test():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    best_model = xgbc_grid_search_cv(X, y, objective='multi:softprob', num_class=30)
    best_model.fit(X_train, y_train)
    y_label_train = best_model.predict(X_train)
    y_label_test = best_model.predict(X_test)
    y_pred_train = best_model.predict_proba(X_train)
    y_pred_test = best_model.predict_proba(X_test)
    print('log loss', log_loss(y_train, y_pred_train))
    print('log loss', log_loss(y_test, y_pred_test))
    print(confusion_matrix(y_train, y_label_train))
    print(confusion_matrix(y_test, y_label_test))
    print(classification_report(y_train, y_label_train))
    print(classification_report(y_test, y_label_test))

# test()
