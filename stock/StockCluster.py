# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.font_manager import FontProperties
import numpy as np
from sklearn import cluster, covariance, manifold
import TushareStore as store

# X(N,M) M为股票数量，N为时间点个数
def fit(X):
    X /= X.std(axis=0)
    # 稀疏的可逆协方差估计
    edge_model = covariance.GraphicalLassoCV(cv=5)
    edge_model.fit(X)
    # #############################################################################
    # edge_model.covariance_ 协方差
    # 聚类
    _, labels = cluster.affinity_propagation(edge_model.covariance_)
    # #############################################################################
    # 非线性降维
    node_position_model = manifold.LocallyLinearEmbedding(n_components=2, eigen_solver='dense',
                                                          n_neighbors=labels.max() + 1)
    embedding = node_position_model.fit_transform(X.T).T
    partial_correlations = edge_model.precision_.copy()
    return labels, partial_correlations, embedding


# names 股票名称列表 [name1,name2,name3]
# labels 股票对应的分类 [cluster1,cluster2,cluster3]
# partial_correlations 偏相关分析 (N*N)
# embedding 映射到平面坐标 [[x1,x2,x3],[y1,y2,y3]]
def draw(names, labels, partial_correlations, embedding):
    n_labels = labels.max()
    # Visualization
    plt.figure(1, facecolor='w', figsize=(10, 8))
    plt.clf()
    ax = plt.axes([0., 0., 1., 1.])
    plt.axis('off')

    # Display a graph of the partial correlations
    # partial_correlations = edge_model.precision_.copy()
    d = 1 / np.sqrt(np.diag(partial_correlations))
    partial_correlations *= d
    partial_correlations *= d[:, np.newaxis]
    non_zero = (np.abs(np.triu(partial_correlations, k=1)) > 0.02)

    # Plot the nodes using the coordinates of our embedding
    plt.scatter(embedding[0], embedding[1], s=100 * d ** 2, c=labels,
                cmap=plt.cm.nipy_spectral)

    # Plot the edges
    start_idx, end_idx = np.where(non_zero)
    # a sequence of (*line0*, *line1*, *line2*), where::
    #            linen = (x0, y0), (x1, y1), ... (xm, ym)
    segments = [[embedding[:, start], embedding[:, stop]]
                for start, stop in zip(start_idx, end_idx)]
    values = np.abs(partial_correlations[non_zero])
    lc = LineCollection(segments,
                        zorder=0, cmap=plt.cm.hot_r,
                        norm=plt.Normalize(0, .7 * values.max()))
    lc.set_array(values)
    lc.set_linewidths(15 * values)
    ax.add_collection(lc)
    font = FontProperties(fname='/System/Library/Fonts/PingFang.ttc', size=10)
    # Add a label to each node. The challenge here is that we want to
    # position the labels to avoid overlap with other labels
    for index, (name, label, (x, y)) in enumerate(
            zip(names, labels, embedding.T)):

        dx = x - embedding[0]
        dx[index] = 1
        dy = y - embedding[1]
        dy[index] = 1
        this_dx = dx[np.argmin(np.abs(dy))]
        this_dy = dy[np.argmin(np.abs(dx))]
        if this_dx > 0:
            horizontalalignment = 'left'
            x = x + .002
        else:
            horizontalalignment = 'right'
            x = x - .002
        if this_dy > 0:
            verticalalignment = 'bottom'
            y = y + .002
        else:
            verticalalignment = 'top'
            y = y - .002
        plt.text(x, y, name, size=10,
                 horizontalalignment=horizontalalignment,
                 verticalalignment=verticalalignment,
                 bbox=dict(facecolor='w',
                           edgecolor=plt.cm.nipy_spectral(label / float(n_labels)),
                           alpha=.6), fontproperties=font)

    plt.xlim(embedding[0].min() - .15 * embedding[0].ptp(),
             embedding[0].max() + .10 * embedding[0].ptp(), )
    plt.ylim(embedding[1].min() - .03 * embedding[1].ptp(),
             embedding[1].max() + .03 * embedding[1].ptp())

    plt.show()


def loadData(start_code='', end_code='', start_date='', end_date=''):
    quotes = []
    names = []
    stock_basics = store.get_basic_stock(start=start_code, end=end_code)
    for index, stock_basic in stock_basics.iterrows():
        code, name = stock_basic['ts_code'], stock_basic['name']
        data = store.get_chart_data_from_db(code, start_date, end_date, append_ind=False)
        if len(data) == 251:
            quotes.append(data[['open', 'close']])
            # quotes.append((data.close - data.open).T.values)
            names.append(name)
            print code, name
    close_prices = np.vstack([q['close'].values for q in quotes])
    open_prices = np.vstack([q['open'].values for q in quotes])
    variation = close_prices - open_prices
    return variation.copy().T, np.array(names)


X, names = loadData(start_code='6000', end_code='6001', start_date='20180101')
labels, partial_correlations, embedding = fit(X)
for i in range(labels.max() + 1):
    print('Cluster %i: %s' % ((i + 1), ', '.join(names[labels == i])))
draw(names, labels, partial_correlations, embedding)
