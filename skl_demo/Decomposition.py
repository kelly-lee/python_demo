# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.datasets import fetch_lfw_people
from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


def pca(data, n_components=None, svd_solver='auto'):
    """
    :param data:
    :param n_components: 可通过plot_pca_evr获得最优值
    :param svd_solver:
    :return:
    """
    if n_components is None:
        n_components = 'mle'
    if n_components < 1:
        svd_solver = 'full'
    pca = PCA(n_components=n_components, svd_solver=svd_solver)
    return pca, pca.fit_transform(data)


def plot_pca_evr(data, pca=None):
    if pca is None:
        pca = PCA()
    plt.plot(np.cumsum(pca.fit(data).explained_variance_ratio_))
    plt.show()


def plot_score_pca_rfc(X, y, range):
    """
    PCA中n_components调参学习曲线
    :param X:
    :param y:
    :param range:
    :return:
    """
    scores = []
    for n in range:
        p, X_d = pca(X, n)
        rfc = RandomForestClassifier(n_estimators=10, random_state=0)
        score = cross_val_score(rfc, X_d, y, cv=5).mean()
        print n, score, X_d.shape
        scores.append(score)
    plt.plot(range, scores)
    plt.show()


def plot_score_pca_knn(X, y, range):
    """
    KNN中n调参学习曲线
    :param X:
    :param y:
    :param range:
    :return:
    """
    scores = []
    for n in range:
        p, X_d = pca(X, 26)
        knn = KNeighborsClassifier(n)
        score = cross_val_score(knn, X_d, y, cv=5).mean()
        print n, score, X_d.shape
        scores.append(score)
    plt.plot(range, scores)
    plt.show()


def best_score_pca_knn(X, y):
    p, X_d = pca(X, 26)
    knn = KNeighborsClassifier(5)
    score = cross_val_score(knn, X_d, y, cv=5).mean()
    print score  # 0.9709756641532662


def best_score_pca_rfc(X, y):
    p, X_d = pca(X, 26)
    rfc = RandomForestClassifier(n_estimators=100, random_state=0)
    score = cross_val_score(rfc, X_d, y, cv=5).mean()
    print score  # 0.9466909208217148


def plot_score_pca_rfc(X, y, range):
    scores = []
    for n in range:
        p, X_d = pca(X, n)
        rfc = RandomForestClassifier(n_estimators=10, random_state=0)
        score = cross_val_score(rfc, X_d, y, cv=5).mean()
        print n, score, X_d.shape
        scores.append(score)
    plt.plot(range, scores)
    plt.show()


def test_pca_iris():
    iris = load_iris()
    X = iris.data
    y = iris.target
    # pca, X_d = pca(X, n_components=2)
    # pca, X_d = pca(X)
    p, X_d = pca(X, 2)
    print X_d
    # 降维后特征向量信息量大小
    print p.explained_variance_
    # 降维后特征向量占原始信息量百分比
    print np.cumsum(p.explained_variance_ratio_)
    # 分解SVD中的V矩阵
    print p.components_
    for i in np.unique(y):
        plt.scatter(X_d[y == i, 0], X_d[y == i, 1])
    plt.show()


def test_pac_digits():
    digits = load_digits()
    print digits.data.shape
    print digits.images.shape
    print digits.target.shape

    random_state = np.random.RandomState(0)
    digits_nosiy = random_state.normal(digits.data, 2)
    p, digits_d = pca(digits_nosiy, 0.5)
    print p.components_.shape
    fig, axes = plt.subplots(3, 5
                             , figsize=(5, 3)
                             , subplot_kw={"xticks": [], "yticks": []})
    for col in range(5):
        axes[0, col].imshow(digits.images[col, :, :], cmap='gray')
        axes[1, col].imshow(p.components_[col].reshape(8, 8), cmap='gray')
        axes[2, col].imshow(p.inverse_transform(digits_d)[col].reshape(8, 8), cmap='gray')
    plt.show()


def test_pca_faces():
    # ~\scikit_learn_data\fw_home
    faces = fetch_lfw_people(min_faces_per_person=60)
    print faces.images.shape  # (1348, 62, 47)
    print faces.data.shape  # (1348, 2914)
    print faces.target.shape  # (1348,)

    p, faces_d = pca(faces, 150)
    print p.components_.shape  # (400, 2914)
    print faces_d.shape  # (1348, 400)

    fig, axes = plt.subplots(3, 20
                             , figsize=(10, 3)
                             , subplot_kw={"xticks": [], "yticks": []})
    for col in range(20):
        axes[0, col].imshow(faces.images[col, :, :], cmap='gray')
        axes[1, col].imshow(p.components_[col].reshape(62, 47), cmap='gray')
        axes[2, col].imshow(p.inverse_transform(faces_d)[col].reshape(62, 47), cmap='gray')
    plt.show()


def test_pac_digit_reg():
    df = pd.read_csv("digit_recognizor.csv")
    print df.info()
    print df.head()
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]
    # plot_pca_evr(X)
    # plot_score_pca_rfc(X,y,np.arange(50, 400, 50))
    # plot_score_pca_rfc(X, y, np.arange(10, 100, 10))
    # plot_score_pca_rfc(X, y, np.arange(20, 30, 1))
    # best_score_pca_rfc(X,y)

    # plot_score_pca_knn(X, y, np.arange(1, 10, 1))

    best_score_pca_knn(X, y)


if __name__ == '__main__':
    # test_pca_iris()
    # test_pca_faces()
    # test_pac_digits()
    test_pac_digit_reg()
