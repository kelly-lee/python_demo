# -*- coding: utf-8 -*-

# 贷前触碰规则
# M1+ overdue 30days 信用风险
# M3+ overdue 90days 欺诈风险
# 有时间周期 6个月后才能判断好用户或坏用户
# 真实好坏样本比例  1/10 ~ 1/30
# 评分卡 降采样
# 随机森林 不用做降采样，过采样
# 过采样，不超过100个，减少观察周期，不严格的欺诈定义
# 逻辑回归 样本偏差不能小于 1/10
# 评分卡 逻辑回归 变量不超过20个
# 600 分 1/50 逾期率  650分 1/25 逾期率 芝麻信用 300~900分
# 稳定性  特征的稳定性，看每季度 均值和方差  集成  稳定运行3~6个月

# 122列
# 删数据
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import RankingCard
import FeatureSelection
import DataPreproceing
import scikitplot as skplt

import xgboost as xgb
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor


def same_info(data, y):
    col_val_p = pd.DataFrame()
    for column in data.columns:
        if column == y:
            continue
        col_val_count = data[column].value_counts()
        col_val_count_max = col_val_count.max().astype(float)
        col_val_count_max_item = col_val_count[col_val_count == col_val_count_max]
        col_val_count_max_item_name = col_val_count_max_item.index.values[0]
        p = col_val_count_max / len(data)
        try:
            p_c = data.groupby([column, y]).size().unstack()
            p_c = p_c.loc[col_val_count_max_item_name, :] / p_c.sum()
            col_val_p = col_val_p.append([[column, col_val_count.index.values, col_val_count_max_item_name,
                                           p, p_c]])
        except:
            print 'error', column
    col_val_p.columns = ['feature', 'items', 'max_count_item', 'max_count_pct', 'max_count_pct_by_y']
    col_val_p.index = col_val_p['feature']
    col_val_p.drop(columns=['feature'], inplace=True)
    return col_val_p.sort_values(by='max_count_pct', ascending=False).head(20)


def info(data):
    print data.shape
    print data.dtypes.value_counts()
    # missing_pct = data.isnull().sum() / len(df)
    # print missing_pct.sort_values(ascending=False)
    # same_info_df = same_info(data, 'loan_status')
    # same_info_df.to_csv('LC_SAME_VAL_2016Q3.csv')
    print data.select_dtypes(include=['O']).describe().T
    # print data.select_dtypes(include=['float']).describe().T
    # print data.select_dtypes(include=['int']).describe().T


def drop_same_info(data, y, threshold):
    same_info_df = same_info(data, y)
    drop_columns = same_info_df[same_info_df['max_count_pct'] >= threshold].index
    print 'drop columns by same val', drop_columns
    data.drop(columns=drop_columns, inplace=True)


def drop(data):
    # 删除重复
    df.drop_duplicates(inplace=True)
    # 删除空行
    df.dropna(axis=0, how='all', inplace=True)
    # 删除空列
    df.dropna(axis=1, how='all', inplace=True)


def drop_high_missing_pct(data, threshold):
    # 基本上删除的第二共同贷款人信息
    missing_pct = data.isnull().sum() / len(df)
    drop_columns = missing_pct[missing_pct > threshold].index
    print 'drop columns by high missing pct', drop_columns
    # thresh_count = len(data) * threshold  # 设定阀值
    # data.dropna(thresh=thresh_count, axis=1, inplace=True)
    data.drop(columns=drop_columns, inplace=True)


def encode_target(data):
    data.loan_status.replace('Fully Paid', int(1), inplace=True)
    data.loan_status.replace('Current', int(1), inplace=True)
    data.loan_status.replace('Late (16-30 days)', int(0), inplace=True)
    data.loan_status.replace('Late (31-120 days)', int(0), inplace=True)
    data.loan_status.replace('Charged Off', int(0), inplace=True)
    data.loan_status.replace('In Grace Period', np.nan, inplace=True)
    data.loan_status.replace('Default', np.nan, inplace=True)
    data.dropna(subset=['loan_status'], inplace=True)


def drop_features(data, features):
    data.drop(features, 1, inplace=True)


def drop_high_type(data, threshold):
    type_desc = data.select_dtypes(include=['O']).describe().T
    drop_columns = type_desc[type_desc['unique'] > threshold].index
    # print type_desc.sort_values(by='unique', ascending=False)
    # print drop_columns
    df.drop(columns=drop_columns, inplace=True)


def drop_biz(data):
    # data.drop('revol_util', 1, inplace=True)
    # data.drop('zip_code', 1, inplace=True)
    # data.drop('earliest_cr_line', 1, inplace=True)
    # data.drop('addr_state', 1, inplace=True)
    # int_rate,sub_grade
    # data.drop('purpose', 1, inplace=True)
    # last_credit_pull_d,last_pymnt_d
    # data.drop('title', 1, inplace=True)
    # data.drop('term', 1, inplace=True)
    # data.drop('issue_d', 1, inplace=True)
    # df.drop('',1,inplace=True)
    # 贷后相关的字段

    data.drop(['out_prncp', 'out_prncp_inv', 'total_pymnt',
               'total_pymnt_inv', 'total_rec_prncp', 'grade', 'sub_grade'], 1, inplace=True)
    data.drop(['total_rec_int', 'total_rec_late_fee',
               'recoveries', 'collection_recovery_fee',
               'collection_recovery_fee'], 1, inplace=True)
    data.drop(['last_pymnt_d', 'last_pymnt_amnt',
               'next_pymnt_d', 'last_credit_pull_d'], 1, inplace=True)
    data.drop(['policy_code'], 1, inplace=True)


# term                       99120      2           36 months  73898
# int_rate                   99120     49              11.49%   8316
# revol_util                 99060   1086                  0%    440
# emp_length                 93198     11           10+ years  34219
def encode(data, features):
    # 不是0-9的字符删除
    for feature in features:
        data[feature].replace(to_replace='[^0-9]+', value='', inplace=True, regex=True)
        data[feature].fillna(value=0, inplace=True)
        data[feature] = df[feature].astype(int)


def draw_bar(data, features):
    row = 4
    col = 5
    fig = plt.figure(figsize=(16, 8))
    i = 0
    for feature in features:
        i += 1
        ax = fig.add_subplot(row, col, i)
        p = data.groupby([feature, 'loan_status']).size().unstack()
        print p
        p = p / p.sum()
        print p.iloc[:, 0], p.index
        ax.bar(p.index, height=p.iloc[:, 0], width=0.45, alpha=0.8, color='red', label="bad")
        # plt.bar(p.columns, height=p.iloc[0, :], width=0.45, alpha=0.8, color='red', label="bad")
        # plt.bar(p.columns, height=p.iloc[1, :], width=0.45, alpha=0.8, color='green', label="good", bottom=p.iloc[0, :])
        ax.legend(fontsize=9, ncol=3)
        ax.set_xticks(p.index)
        ax.set_xticklabels(p.index)
        ax.set_ylabel(feature)
    plt.show()


def train_test(data):
    y = data.loan_status
    X = data.drop('loan_status', 1, inplace=False)
    # X['title'].fillna('Other', inplace=True)
    X = pd.get_dummies(X)
    # print X.columns
    print X.shape

    X.fillna(0.0, inplace=True)
    X.fillna(0, inplace=True)
    # print X.select_dtypes(include=['O']).describe().T
    # scaler, X = DataPreproceing.normalization(X)

    return train_test_split(X, y, test_size=0.3, random_state=123)


def train_gbr(X_train, X_test, y_train, y_test):
    est = GradientBoostingRegressor(min_samples_split=100, n_estimators=100,
                                    learning_rate=0.1, max_depth=3, random_state=0, loss='ls')
    est.fit(X_train, y_train)
    print est.score(X_test, y_test)

    feature_importance = est.feature_importances_
    print np.argsort(feature_importance)
    print X_train.columns[np.argsort(feature_importance)]


def train_lr(X_train, X_test, y_train, y_test):
    lr = LogisticRegression(penalty='l2', C=100, solver='newton-cg', max_iter=25)
    lr.fit(X_train, y_train)
    print lr.score(X_test, y_test)
    vali_proba_df = pd.DataFrame(lr.predict_proba(X_test))
    # skplt.metrics.plot_roc(y_test, vali_proba_df,
    #                        plot_micro=False, figsize=(6, 6),
    #                        plot_macro=False)
    skplt.metrics.plot_ks_statistic(y_test, vali_proba_df)
    plt.show()


def train_xgboost(X_train, X_test, y_train, y_test):
    # XGBoost
    clf2 = xgb.XGBClassifier(n_estimators=50, max_depth=1,
                             learning_rate=0.01, subsample=0.8, colsample_bytree=0.3, scale_pos_weight=3.0,
                             silent=True, nthread=-1, seed=0, missing=None, objective='binary:logistic',
                             reg_alpha=1, reg_lambda=1,
                             gamma=0, min_child_weight=1,
                             max_delta_step=0, base_score=0.5)
    clf2.fit(X_train, y_train)
    print clf2.score(X_test, y_test)
    # test_pd2 = pd.DataFrame()
    # test_pd2['predict'] = clf2.predict(X_test)
    # test_pd2['label'] = y_test
    # # print compute_ks(test_pd[['label', 'predict']])
    feature_importance = clf2.feature_importances_
    feature_importance = pd.Series(data=feature_importance[np.argsort(feature_importance)],
                                   index=X_train.columns[np.argsort(feature_importance)])
    print feature_importance.sort_values(ascending=False).head(10)
    # 0.9732521093722648
    # 0.9732521093722648
    # 0.244,0.64
    # 0.255,0.66
    # 0.264,0.66  删除相关性变量
    fig = plt.figure(figsize=(8, 4))
    vali_proba_df = pd.DataFrame(clf2.predict_proba(X_test))
    ax = fig.add_subplot(1, 2, 1)
    skplt.metrics.plot_ks_statistic(y_test, vali_proba_df, ax=ax)
    ax = fig.add_subplot(1, 2, 2)
    skplt.metrics.plot_roc(y_test, vali_proba_df, ax=ax,
                           plot_micro=False, figsize=(6, 6),
                           plot_macro=False)
    plt.show()

    # Top Ten
    # feature_importance = clf2.feature_importances_
    # feature_importance = 100.0 * (feature_importance / feature_importance.max())
    #
    # indices = np.argsort(feature_importance)[-10:]
    # plt.barh(np.arange(10), feature_importance[indices], color='dodgerblue', alpha=.4)
    # plt.yticks(np.arange(10 + 0.25), np.array(X_train.columns)[indices])
    # _ = plt.xlabel('Relative importance'), plt.title('Top Ten Important Variables')
    # plt.show()


if __name__ == '__main__':
    df = pd.read_csv("LC_2016Q3.csv", low_memory=False)
    # [ 'emp_title',  'title',
    #  'initial_list_status',
    # drop_biz(df)
    # 时间类的删除: 最早授信日，最近授信日，最后还款日，下一个还款日，放款日
    drop_features(df, ['earliest_cr_line', 'last_credit_pull_d', 'issue_d', 'next_pymnt_d', 'last_pymnt_d'])
    drop_features(df, ['addr_state', 'zip_code'])
    # 删除相关性变量
    drop_features(df, ['funded_amnt', 'funded_amnt_inv', 'installment', 'title'])
    # 贷后字段
    # drop_features(df, ['recoveries', 'collection_recovery_fee'])
    drop_features(df, ['total_pymnt', 'total_rec_prncp', 'total_pymnt_inv', 'total_rec_int', 'total_rec_late_fee'])
    # drop_features(df, ['last_pymnt_amnt'])
    # # 2字段值一样
    # drop_features(df, ['out_prncp', 'out_prncp_inv'])
    drop_features(df, ['grade', 'sub_grade'])

    # print info(df)

    # 删除空行空列重复数据
    drop(df)
    # print info(df)
    drop_high_missing_pct(df, threshold=0.9)
    encode_target(df)
    drop_same_info(df, 'loan_status', threshold=0.9)
    encode(df, features=['emp_length', 'revol_util', 'term', 'int_rate'])
    drop_high_type(df, 49)
    print info(df)

    # 这一步会降分
    # 'next_pymnt_d', 'last_credit_pull_d',

    # 负样本中取值单一的
    # drop_features(df, ['pymnt_plan', 'application_type', 'issue_d'])
    # drop_features(df, ['title', 'purpose'])

    print df.shape

    # print df['acc_open_past_24mths'].value_counts()

    # for key in ['int_rate', 'acc_open_past_24mths', 'emp_length', 'open_rv_24m', 'num_tl_op_past_12m', 'dti']:
    #     # RankingCard.plot_iv(df, key, 'loan_status', 10)
    #     bins = RankingCard.get_bins(df, key, 'loan_status', 5, 20)
    #     woe = RankingCard.get_woe(df, key, 'loan_status', bins)
    #     df[key] = pd.cut(df[key], bins).map(woe)
    #     print df[key].value_counts()

    # draw_bar(df, features=['home_ownership', 'verification_status',
    #                        'initial_list_status',
    #                        'grade', 'sub_grade', 'issue_d', 'purpose', 'title',
    #                        'last_pymnt_d'])
    # print df.shape
    # drop_high_type(df, 49)
    # print df.shape

    # print (df.isnull().sum() / len(df)).sort_values()
    #
    # print df.select_dtypes(include=['O']).describe().T
    # print df['home_ownership'].value_counts()  # 哑变量  抵押45105,租金38253,拥有11846,任何6个
    # print df['verification_status'].value_counts()  # 哑变量
    # print df['loan_status'].value_counts()  # 哑变量
    # print df['pymnt_plan'].value_counts()  # 哑变量
    # print df['initial_list_status'].value_counts()  # 哑变量
    # print df['application_type'].value_counts()  # 哑变量
    #
    # df = df[['int_rate','acc_open_past_24mths','verification_status','title','loan_status']]
    X_train, X_test, y_train, y_test = train_test(df)
    # train_gbr(X_train, X_test, y_train, y_test)
    # train_lr(X_train, X_test, y_train, y_test)
    train_xgboost(X_train, X_test, y_train, y_test)
