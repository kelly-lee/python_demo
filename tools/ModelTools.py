#!/usr/bin/env python
# -*- coding: utf-8 -*-
# coding: utf-8
import sys
import numpy as np
from lightgbm import LGBMRegressor
from mlxtend.regressor import StackingCVRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor


def report(y, y_pred, type='train'):
    cm = confusion_matrix(y, y_pred)
    print('%s confusion matrix:\n %s' % (type, cm))
    cr = classification_report(y, y_pred)
    print('%s classification report:\n %s' % cr)
    ras = roc_auc_score(y, y_pred)
    print('%s roc auc: %s' % ras)


def cv_rmse(model, X, y):
    kfolds = KFold(n_splits=10, shuffle=True, random_state=42)
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kfolds))
    return (rmse)


def linear_cv_fit(X, y):
    alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
    alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
    e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
    e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]
    kfolds = KFold(n_splits=10, shuffle=True, random_state=42)
    # 二范数rideg岭回归模型
    ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_alt, cv=kfolds))
    # 一范数LASSO收缩模型
    lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, alphas=alphas2, random_state=42, cv=kfolds))
    # 弹性网络模型（结合了ridge和lasso的特点，同时使用了L1和L2作为正则化项)
    elasticnet = make_pipeline(RobustScaler(),
                               ElasticNetCV(max_iter=1e7, alphas=e_alphas, cv=kfolds, l1_ratio=e_l1ratio))
    ridge_cv_rmse = cv_rmse(ridge, X, y)
    lasso_cv_rmse = cv_rmse(lasso, X, y)
    elasticnet_cv_rmse = cv_rmse(elasticnet, X, y)

    print('ridge cv_rmse = %.4f (%.4f)' % (ridge_cv_rmse.mean(), ridge_cv_rmse.std()))
    print('lasso cv_rmse = %.4f (%.4f)' % (lasso_cv_rmse.mean(), lasso_cv_rmse.std()))
    print('elasticnet cv_rmse = %.4f (%.4f)' % (elasticnet_cv_rmse.mean(), elasticnet_cv_rmse.std()))


    elasticnet = elasticnet.fit(X, y)
    lasso = lasso.fit(X, y)
    ridge = ridge.fit(X, y)
    return (lasso, ridge, elasticnet)


def tree_cv_fit(X, y):
    # 定义SVM支持向量机模型
    svr = make_pipeline(RobustScaler(), SVR(C=20, epsilon=0.008, gamma=0.0003, ))
    # 定义GB梯度提升模型（展开到一阶导数）
    gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt',
                                    min_samples_leaf=15, min_samples_split=10, loss='huber', random_state=42)
    # 定义lightgbm模型
    lightgbm = LGBMRegressor(objective='regression',
                             num_leaves=4,
                             learning_rate=0.01,
                             n_estimators=5000,
                             max_bin=200,
                             bagging_fraction=0.75,
                             bagging_freq=5,
                             bagging_seed=7,
                             feature_fraction=0.2,
                             feature_fraction_seed=7,
                             verbose=-1,
                             # min_data_in_leaf=2,
                             # min_sum_hessian_in_leaf=11
                             )
    # 定义xgboost模型（展开到二阶导数）
    xgboost = XGBRegressor(learning_rate=0.01, n_estimators=3460,
                           max_depth=3, min_child_weight=0,
                           gamma=0, subsample=0.7,
                           colsample_bytree=0.7,
                           objective='reg:linear', nthread=-1,
                           scale_pos_weight=1, seed=27,
                           reg_alpha=0.00006)
    svr_cv_rmse = cv_rmse(svr, X, y)
    gbr_cv_rmse = cv_rmse(gbr, X, y)
    lightgbm_cv_rmse = cv_rmse(lightgbm, X, y)
    xgboost_cv_rmse = cv_rmse(xgboost, X, y)
    print('svr cv_rmse = %.4f (%.4f)' % (svr_cv_rmse.mean(), svr_cv_rmse.std()))
    print('gbr cv_rmse = %.4f (%.4f)' % (gbr_cv_rmse.mean(), gbr_cv_rmse.std()))
    print('lightgbm cv_rmse = %.4f (%.4f)' % (lightgbm_cv_rmse.mean(), lightgbm_cv_rmse.std()))
    print('xgboost cv_rmse = %.4f (%.4f)' % (xgboost_cv_rmse.mean(), xgboost_cv_rmse.std()))
    svr = svr.fit(X, y)
    gbr = gbr.fit(X, y)
    lightgbm = lightgbm.fit(X, y)
    xgboost = xgboost.fit(X, y)
    return svr, gbr, xgboost, lightgbm
