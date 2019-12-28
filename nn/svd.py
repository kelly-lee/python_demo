# !/usr/bin/env python
# -*- coding: utf-8 -*-
# coding: utf-8
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys

# reload(sys)
# sys.setdefaultencoding('utf8')
from matplotlib.font_manager import FontProperties

font = FontProperties(fname='/System/Library/Fonts/PingFang.ttc', size=8)
from mittens import GloVe
# 编造文本、分词
import jieba

# 生成字典和词ID
from gensim.corpora import Dictionary
# 共现矩阵
import numpy as np
# 可视化
import matplotlib.pyplot as mp
from sklearn.cluster import KMeans


def co_occurrence_matrix(matrix, ls):
    length = len(ls)
    for i in range(length):
        for j in range(length):
            if i != j:
                matrix[[ls[i]], [ls[j]]] += 1


def test1():
    sentences = [['我吴彦祖', '我张学友'], ['吴彦祖我', '张学友我刘德华吴彦祖'],
                 ['酸奶芝士', '芝士酸奶'], ['芝士蛋糕', '酸奶芝士蛋糕']]
    ls_of_words = [jieba.lcut(sentence) for sentence in sentences]

    dt = Dictionary(ls_of_words).token2id
    ls_of_wids = [[dt[word] for word in words] for words in ls_of_words]

    dimension = len(dt)  # 维数
    matrix = np.matrix([[0] * dimension] * dimension)

    for ls in ls_of_wids:
        co_occurrence_matrix(matrix, ls)
    print(matrix)

    # 奇异值分解（Singular Value Decomposition）
    U, s, Vh = np.linalg.svd(matrix, full_matrices=False)

    # 聚类
    X = -U[:, 0:2]

    labels = KMeans(n_clusters=2).fit(X).labels_
    colors = ('y', 'g')

    mp.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
    for word in dt.keys():
        i = dt[word]
        mp.scatter(X[i, 1], X[i, 0], c=colors[labels[i]], s=400, alpha=0.4)
        mp.text(X[i, 1], X[i, 0], word, ha='center', va='center')
    mp.show()


if __name__ == '__main__':
    # test1()
    docs = [['我吴彦祖我张学友'], ['吴彦祖我张学友我刘德华吴彦祖'],
            ['酸奶芝士芝士酸奶'], ['芝士蛋糕酸奶芝士蛋糕']]
    words = [jieba.lcut(sentence.decode('utf-8', 'ignore')) for doc in docs for sentence in doc]
    dictionary = Dictionary(words)
    corpus = [dictionary.doc2bow(word) for word in words]
    size = len(dictionary.token2id)
    matrix = np.zeros((size, size))
    windows = 2

    for doc_words in words:
        for center_index, center_word in enumerate(doc_words):
            center_id = dictionary.token2id[center_word]
            context_start_index = max(center_index - windows, 0)
            context_end_index = min(center_index + windows + 1, len(doc_words))
            window_words = doc_words[context_start_index:context_end_index]
            print ',,,,,'.join(window_words)
            for context_index in range(context_start_index, context_end_index):
                if center_index != context_index:
                    context_word = doc_words[context_index]
                    context_id = dictionary.token2id[context_word]
                    print center_id, center_word, context_id, context_word
                    matrix[center_id, context_id] += 1
    print matrix
    # 奇异值分解（Singular Value Decomposition）
    U, s, Vh = np.linalg.svd(matrix, full_matrices=False)

    # 聚类
    X = -U[:, 0:2]

    labels = KMeans(n_clusters=2).fit(X).labels_
    colors = ('y', 'g')

    glove_model = GloVe(n=2, max_iter=200)
    embeddings = glove_model.fit(matrix)
    X = embeddings
    mp.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
    for word in dictionary.token2id.keys():
        i = dictionary.token2id[word]
        mp.scatter(X[i, 1], X[i, 0], c=colors[labels[i]], s=400, alpha=0.4)
        mp.text(X[i, 1], X[i, 0], word, ha='center', va='center', fontproperties=font)
    mp.show()
