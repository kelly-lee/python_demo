# -*- coding: utf-8 -*-
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabaz_score
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt


def kmeans_train(data, n_clusters):
    cluster = KMeans(n_clusters=n_clusters, random_state=0)
    y_predict = cluster.fit_predict(data)
    return cluster, y_predict


if __name__ == '__main__':
    X, y = make_blobs(n_samples=500, n_features=2, centers=4, random_state=1)
    print X
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()
    n_clusters = 4
    cluster, y_predict = kmeans_train(X, n_clusters=n_clusters)
    # 中心点坐标
    cluster_centers = cluster.cluster_centers_
    # 聚类结果
    print cluster.labels_
    print calinski_harabaz_score(X, y_predict)
    print davies_bouldin_score(X, y_predict)

    # plt.scatter(X[:, 0], X[:, 1], c=y_predict)
    # plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='x', c='black')
    # plt.show()
