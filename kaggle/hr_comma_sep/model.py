#!/usr/bin/env python
# -*- coding: utf-8 -*-
# coding: utf-8
import sys
import pandas as pd
import numpy as np
from tools import DrawTools
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from tools import ModelTools
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

# ['satisfaction_level', 'last_evaluation', 'number_project',
#        'average_montly_hours', 'time_spend_company', 'Work_accident', 'left',
#        'promotion_last_5years', 'sales', 'salary'],
data = pd.read_csv('data/HR_comma_sep.csv')
col = ['satisfaction_level', 'last_evaluation', 'number_project',
       'average_montly_hours', 'time_spend_company', 'Work_accident', 'promotion_last_5years', 'salary']
data['salary'].replace({'low': 0, 'medium': 1, 'high': 2}, inplace=True)
DrawTools.init()
# DrawTools.displot_mul(data, feature_xs=col, feature_h='left', grid=(3, 3))
plt.show()
data = pd.get_dummies(data)

y = data['left']
X = data.loc[:, data.columns != 'left']
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=0)
model = XGBClassifier()
model = model.fit(Xtrain, ytrain)
ytrain_pred = model.predict(Xtrain)
ytest_pred = model.predict(Xtest)

ModelTools.report(ytrain, ytrain_pred, ytest, ytest_pred)
DrawTools.feature_importance(model, data.columns[data.columns != 'left'])