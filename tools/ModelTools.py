#!/usr/bin/env python
# -*- coding: utf-8 -*-
# coding: utf-8
import sys

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score


def report(ytrain, ytrain_pred, ytest, ytest_pred):
    train_confusion_matrix = confusion_matrix(ytrain, ytrain_pred)
    test_confusion_matrix = confusion_matrix(ytest, ytest_pred)
    print('train confusion matrix:\n %s' % train_confusion_matrix)
    print('test confusion matrix:\n %s' % test_confusion_matrix)
    train_classification_report = classification_report(ytrain, ytrain_pred)
    test_classification_report = classification_report(ytest, ytest_pred)
    print('train classification report:\n %s' % train_classification_report)
    print('test classification repor:\n %s' % test_classification_report)
    train_roc_auc_score = roc_auc_score(ytrain, ytrain_pred)
    test_roc_auc_score = roc_auc_score(ytest, ytest_pred)
    print('train roc auc: %s' % train_roc_auc_score)
    print('test roc auc: %s' % test_roc_auc_score)
