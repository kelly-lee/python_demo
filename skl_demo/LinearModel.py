# -*- coding:UTF-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score

import FeatureSelection
import DataPreproceing
import imblearn
from imblearn.over_sampling import SMOTE
import scikitplot as skplt
import scipy


def log_reg_train(X, y, C, max_iter=10000, penalty='l2'):
    logistic_regression = LogisticRegression(penalty=penalty, C=C, solver='newton-cg', max_iter=max_iter,
                                             multi_class='auto')
    logistic_regression.fit(X, y)
    return logistic_regression


def plot_score_log_reg_c(X, y, range):
    # c = 2050 ,s = 0.963
    scores = []
    for c in range:
        logistic_regression = LogisticRegression(penalty='l2', C=c, solver='newton-cg', max_iter=1000,
                                                 multi_class='auto')
        score = cross_val_score(logistic_regression, X, y, cv=5, scoring='accuracy')
        scores.append(score.mean())
    print range[scores.index(max(scores))], max(scores)
    plt.plot(range, scores)
    plt.show()


def plot_score_log_reg_multi(X, y):
    # c = 2050 ,s = 0.963
    scores = []
    range = ['ovr', 'multinomial']
    for multi_class in range:
        logistic_regression = LogisticRegression(penalty='l2', C=100, solver='newton-cg', max_iter=1000,
                                                 multi_class=multi_class)
        score = cross_val_score(logistic_regression, X, y, cv=5, scoring='accuracy')
        scores.append(score.mean())
    print range[scores.index(max(scores))], max(scores)
    plt.plot(range, scores)
    plt.show()


def plot_score_log_reg_embedded(X, y):
    # c = 2050 ,s = 0.963
    # threshold = 0.514,shape =16
    logistic_regression = log_reg_train(X, y, C=2050, max_iter=27)
    print logistic_regression.n_iter_
    print logistic_regression.coef_
    range = np.linspace(0, logistic_regression.coef_.max(), 20)
    scores = []
    scores_embedded = []
    for threshold in range:
        score = cross_val_score(logistic_regression, X, y, cv=5, scoring='accuracy')
        select_model, X_embedded = FeatureSelection.select_by_model(logistic_regression, X, y, threshold=threshold)
        score_embedded = cross_val_score(logistic_regression, X_embedded, y, cv=5, scoring='accuracy')
        scores.append(score.mean())
        scores_embedded.append(score_embedded.mean())
    print range[scores_embedded.index(max(scores_embedded))], max(scores_embedded)
    plt.plot(range, scores)
    plt.plot(range, scores_embedded)
    plt.show()
    select_model, X_embedded = FeatureSelection.select_by_model(logistic_regression, X, y, threshold=0.514)
    print X_embedded.shape


def test():
    cancer = load_breast_cancer()
    X = cancer.data
    y = cancer.target
    # C 正则化强度的倒数，大于0的浮点数
    l1_train_scores = []
    l2_train_scores = []
    l1_test_scores = []
    l2_test_scores = []
    range = np.linspace(940, 970, 20)
    # l1 c=2660 s=0.953
    # l2 c=716 s=0.961
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    for c in range:
        l1 = LogisticRegression(penalty='l1', C=c, solver='liblinear', max_iter=1000)
        l2 = LogisticRegression(penalty='l2', C=c, solver='liblinear', max_iter=1000)
        l1.fit(X_train, y_train)
        l2.fit(X_train, y_train)
        y_predict_l1_train = l1.predict(X_train)
        y_predict_l1_test = l1.predict(X_test)
        y_predict_l2_train = l2.predict(X_train)
        y_predict_l2_test = l2.predict(X_test)
        l1_train_score = accuracy_score(y_train, y_predict_l1_train)
        l1_test_score = accuracy_score(y_test, y_predict_l1_test)
        l2_train_score = accuracy_score(y_train, y_predict_l2_train)
        l2_test_score = accuracy_score(y_test, y_predict_l2_test)
        # print l1_train_score, l1_test_score
        print c, l2_train_score, l2_test_score
        l1_train_scores.append(l1_train_score)
        l2_train_scores.append(l2_train_score)
        l1_test_scores.append(l1_test_score)
        l2_test_scores.append(l2_test_score)

    plt.plot(range, l1_train_scores, c='red')
    plt.plot(range, l1_test_scores, c='blue')
    plt.plot(range, l2_train_scores, c='green')
    plt.plot(range, l2_test_scores, c='yellow')
    plt.show()


# def get_woe(num_bins):
#     columns = ["min", "max", "count_0", "count_1"]
#     df = pd.DataFrame(num_bins, columns=columns)
#     # df["total"] = df.count_0 + df.count_1
#     # df["percentage"] = df.total / df.total.sum()
#     # df["bad_rate"] = df.count_1 / df.total
#     df["good%"] = df.count_0 / df.count_0.sum()
#     df["bad%"] = df.count_1 / df.count_1.sum()
#     df["woe"] = np.log(df["good%"] / df["bad%"])
#     df["rate"] = df["good%"] - df["bad%"]
#     df["iv"] = df["rate"] * df.woe
#     print df.head()
#     print df["iv"].sum()
#     return df

def get_iv(df):
    rate = df["good%"] - df["bad%"]
    iv = np.sum(rate * df.woe)
    return iv


def iv(bin_data):
    P = bin_data[[0, 1]]
    P = P / P.sum()
    WOE = np.log(P[0] / P[1])
    IV = (P[0] - P[1]) * WOE
    return IV.sum()


def rankingcard():
    # rankingcard_step1()
    # rankingcard_step2()
    # rankingcard_step3()
    # rankingcard_step4()
    rankingcard_step5()


# def get_woe(df, col, y, bins):
#     df = df[[col, y]].copy()
#     df["cut"] = pd.cut(df[col], bins)
#     bins_df = df.groupby("cut")[y].value_counts().unstack()
#     print bins_df
#     woe = bins_df["woe"] = np.log((bins_df[0] / bins_df[0].sum()) / (bins_df[1] / bins_df[1].sum()))
#     return woe

def get_score_card(s1, odd1, s2, odd2, lr, woe_dic, colunms):
    B = (s2 - s1) * np.log(odd2 / odd1)
    A = s1 + B * np.log(odd1)
    print A, B
    base_score = A - B * lr.intercept_
    print 'base_score', base_score
    for i, key in enumerate(colunms):
        score = woe_dic[key] * (-B * lr.coef_[0][i])
        print i, key, score


def get_woe_dic(train, bin_dic):
    woe_dic = {}
    for key in bin_dic.keys():
        bins = bin_dic[key]
        woe = get_woe(train, key, 'SeriousDlqin2yrs', bins)
        woe_dic[key] = woe
    return woe_dic


def get_bin_dic(train):
    bins_dic = {
        "RevolvingUtilizationOfUnsecuredLines": get_bins(train, 'RevolvingUtilizationOfUnsecuredLines',
                                                         'SeriousDlqin2yrs', 9, 20)
        , "age": get_bins(train, 'age', 'SeriousDlqin2yrs', 14, 20)
        , "NumberOfTime30-59DaysPastDueNotWorse": [0, 1, 2, 13]
        , "DebtRatio": get_bins(train, 'DebtRatio', 'SeriousDlqin2yrs', 7, 20)
        , "MonthlyIncome": get_bins(train, 'MonthlyIncome', 'SeriousDlqin2yrs', 8, 20)
        , "NumberOfOpenCreditLinesAndLoans": [0, 4.626536, 5.939805, 7.228987, 8.873983, 58]
        , "NumberOfTimes90DaysLate": [0, 1, 2, 17]
        , "NumberRealEstateLoansOrLines": [0, 1, 2, 4, 54]
        , "NumberOfTime60-89DaysPastDueNotWorse": [0, 1, 2, 8]
        , "NumberOfDependents": [0, 1, 2, 3]}
    for key in bins_dic.keys():
        bins = bins_dic[key]
        bins[0], bins[-1] = -np.inf, np.inf
    return bins_dic


def get_woe(data, a, c, bins):
    data_copy = data[[a, c]]
    data_copy.loc[:, a] = pd.cut(data_copy[a], bins)
    woe_data = data_copy.groupby([a, c]).size().unstack()
    P = woe_data[[0, 1]]
    P = P / P.sum()
    WOE = np.log(P[0] / P[1])
    return WOE


def get_bins(data, a, c, bins, init_bins):
    bin_data = get_bin_data(data, a, c, init_bins)
    while (len(bin_data) > bins):
        bin_merge(bin_data)
    return sorted(set(bin_data["min"]).union(bin_data["max"]))


def get_bin_data(data, a, c, bins):
    data_copy = data[[a, c]]
    data_copy.loc[:, a], retbins = pd.qcut(data_copy[a], retbins=True, q=bins)

    bin_data = data_copy.groupby([a, c]).size().unstack()
    bin_data.index = np.arange(0, len(bin_data))
    # bin_data['min'] = retbins[0:-1]
    # bin_data['max'] = retbins[1:]
    # return bin_data
    min = pd.Series(retbins[0:-1], index=bin_data.index, name='min')
    max = pd.Series(retbins[1:], index=bin_data.index, name='max')
    return pd.concat([min, max, bin_data], axis=1)


def bin_merge(bin_data):
    chi2_data = pd.concat([bin_data, bin_data.iloc[:, 2:].shift(1)], axis=1)
    chi2_p = chi2_data.apply(
        lambda x: scipy.stats.chi2_contingency([x.values[2:4], x.values[4:6]])[1], axis=1)
    # chi2_data = chi2_data.fillna(0)
    # print chi2_data
    idx = chi2_p.idxmax()
    bin_data.iloc[idx, :] = [
        np.nanmin([bin_data.iat[idx, 0], bin_data.iat[idx - 1, 0]])
        , np.nanmax([bin_data.iat[idx, 1], bin_data.iat[idx - 1, 1]])
        , np.nansum([bin_data.iat[idx, 2], bin_data.iat[idx - 1, 2]])
        , np.nansum([bin_data.iat[idx, 3], bin_data.iat[idx - 1, 3]])]
    bin_data.drop([idx - 1], inplace=True)
    bin_data.index = np.arange(0, len(bin_data))


def plot_iv(data, a, c, init_bins):
    bin_data = get_bin_data(data, a, c, init_bins)
    ivs = []
    bins = []
    ivs.append(iv(bin_data))
    bins.append(len(bin_data))
    while (len(bin_data) > 2):
        bin_merge(bin_data)
        ivs.append(iv(bin_data))
        bins.append(len(bin_data))
    plt.plot(bins, ivs)
    plt.xticks(bins)
    plt.ylabel(a)
    plt.show()


def rankingcard_step5():
    train = pd.read_csv('rankingcard_train.csv')
    test = pd.read_csv('rankingcard_test.csv')
    bin_dic = get_bin_dic(train)
    woe_dic = get_woe_dic(train, bin_dic)
    for key in bin_dic.keys():
        bins = bin_dic[key]
        woe = woe_dic[key]
        train[key] = pd.cut(train[key], bins).map(woe)
        test[key] = pd.cut(test[key], bins).map(woe)
    X_train = train.loc[:, train.columns != 'SeriousDlqin2yrs']
    y_train = train['SeriousDlqin2yrs']
    X_test = test.loc[:, test.columns != 'SeriousDlqin2yrs']
    y_test = test['SeriousDlqin2yrs']
    lr = LogisticRegression(penalty='l2', C=100, solver='newton-cg', max_iter=25)
    lr.fit(X_train, y_train)
    vali_proba_df = pd.DataFrame(lr.predict_proba(X_test))
    skplt.metrics.plot_roc(y_test, vali_proba_df,
                           plot_micro=False, figsize=(6, 6),
                           plot_macro=False)
    plt.show()
    get_score_card(600, 1.0 / 60, 620, 1.0 / 30, lr, woe_dic, X_train.columns)


def rankingcard_step4():
    train = pd.read_csv('rankingcard_train.csv')
    cols = ['RevolvingUtilizationOfUnsecuredLines', 'age', 'DebtRatio', 'MonthlyIncome']
    for col in cols:
        plot_iv(train, col, 'SeriousDlqin2yrs', 20)


def rankingcard_step3():
    # 上采用，处理样本不均衡问题
    df = pd.read_csv('rankingcard_2.csv')
    X = df.iloc[:, df.columns != 'SeriousDlqin2yrs']
    y = df.iloc[:, df.columns == 'SeriousDlqin2yrs']
    smote = SMOTE(random_state=0)
    X, y = smote.fit_sample(X, y)
    # 拆分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
    train = pd.concat([pd.DataFrame(y_train), pd.DataFrame(X_train)], axis=1)
    test = pd.concat([pd.DataFrame(y_test), pd.DataFrame(X_test)], axis=1)
    train.columns = df.columns
    test.columns = df.columns
    train.to_csv('rankingcard_train.csv', index=False)
    test.to_csv('rankingcard_test.csv', index=False)


def rankingcard_step2():
    # 描述性统计处理差异
    df = pd.read_csv('rankingcard_1.csv')
    print df.describe([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]).T
    df = df[df['age'] > 0]
    df = df[df['NumberOfTimes90DaysLate'] < 90]
    df.to_csv('rankingcard_2.csv', index=False)


def rankingcard_step1():
    df = pd.read_csv('rankingcard.csv', index_col=0)
    # 1.去重
    df.drop_duplicates(inplace=True)
    # 2.填补缺失值
    # 家属人数用均值填充
    DataPreproceing.fill_mean(df['NumberOfDependents'].values, copy=False)
    # 收入用随机森林填充
    df.iloc[:, df.columns != 'SeriousDlqin2yrs'] = DataPreproceing.fill_rfr(
        df.iloc[:, df.columns != 'SeriousDlqin2yrs'].values,
        df['SeriousDlqin2yrs'].values)
    df.to_csv('rankingcard_1.csv', index=False)


if __name__ == '__main__':
    # cancer = load_breast_cancer()
    # X = cancer.data
    # y = cancer.target
    # range = np.arange(10, 3000, 50)
    # plot_score_log_reg_c(X, y, range)
    # plot_score_log_reg_embedded(X, y)
    # iris = load_iris()
    # X = iris.data
    # y = iris.target
    # plot_score_log_reg_multi(X, y)
    rankingcard()


def graphforbestbin(DF, X, Y, n=5, q=20, graph=True):
    """
    自动最优分箱函数,基于卡方检验的分箱
    参数:
    DF: 需要输入的数据
    X: 需要分箱的列名
    Y: 分箱数据对应的标签 Y 列名 n: 保留分箱个数
    q: 初始分箱的个数
    graph: 是否要画出IV图像
    区间为前开后闭 (]
    """
    DF = DF[[X, Y]].copy()
    DF["qcut"], bins = pd.qcut(DF[X], retbins=True, q=q, duplicates="drop")
    coount_y0 = DF.loc[DF[Y] == 0].groupby(by="qcut").count()[Y]
    coount_y1 = DF.loc[DF[Y] == 1].groupby(by="qcut").count()[Y]
    # num_bins = [*zip(bins, bins[1:], coount_y0, coount_y1)]
    num_bins = [(bins, bins[1:], coount_y0, coount_y1)]
    for i in range(q):
        if 0 in num_bins[0][2:]:
            num_bins[0:2] = [(
                num_bins[0][0],
                num_bins[1][1],
                num_bins[0][2] + num_bins[1][2],
                num_bins[0][3] + num_bins[1][3])]
        continue
    for i in range(len(num_bins)):
        if 0 in num_bins[i][2:]:
            num_bins[i - 1:i + 1] = [(
                num_bins[i - 1][0],
                num_bins[i][1],
                num_bins[i - 1][2] + num_bins[i][2],
                num_bins[i - 1][3] + num_bins[i][3])]
            break
        else:
            break

# def test2():
#     IV = []
#     axisx = []
#     while len(num_bins) > n:
#         pvs = []
#         for i in range(len(num_bins) - 1):
#             x1 = num_bins[i][2:]
#             x2 = num_bins[i + 1][2:]
#             pv = scipy.stats.chi2_contingency([x1, x2])[1]
#             pvs.append(pv)
#         i = pvs.index(max(pvs))
#         num_bins[i:i + 2] = [(
#             num_bins[i][0],
#             num_bins[i + 1][1],
#             num_bins[i][2] + num_bins[i + 1][2],
#             num_bins[i][3] + num_bins[i + 1][3])]
#         bins_df = pd.DataFrame(get_woe(num_bins))
#         axisx.append(len(num_bins))
#         IV.append(get_iv(bins_df))
#     if graph:
#         plt.figure()
#         plt.plot(axisx, IV)
#         plt.xticks(axisx)
#         plt.show()

# model_data.columns
# for i in model_data.columns[1:-1]:
#     print(i)
#     graphforbestbin(model_data, i, "SeriousDlqin2yrs", n=2, q=20)
# auto_col_bins = {"RevolvingUtilizationOfUnsecuredLines": 6,
#                  "age": 5,
#                  "DebtRatio": 4,
#                  "MonthlyIncome": 3,
#                  "NumberOfOpenCreditLinesAndLoans": 5}
# # 不能使用自动分箱的变量
# hand_bins = {"NumberOfTime30-59DaysPastDueNotWorse": [0, 1, 2, 13]
#     , "NumberOfTimes90DaysLate": [0, 1, 2, 17]
#     , "NumberRealEstateLoansOrLines": [0, 1, 2, 4, 54]
#     , "NumberOfTime60-89DaysPastDueNotWorse": [0, 1, 2, 8]
#     , "NumberOfDependents": [0, 1, 2, 3]}
# # 保证区间覆盖使用 np.inf替换最大值,用-np.inf替换最小值
# hand_bins = {k: [-np.inf, *v[:-1], np.inf] for k, v in hand_bins.items()}
# 接下来对所有特征按照选择的箱体个数和手写的分箱范围进行分箱:
#
# return bins_df
