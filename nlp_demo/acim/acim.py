#!/usr/bin/env python
# -*- coding: utf-8 -*-
# coding: utf-8
import sys, codecs
import jieba
import re
from collections import Counter
import numpy as np
from gensim.models import word2vec
from gensim.models import Word2Vec
from gensim import corpora, models, similarities
import logging
import numpy as np
from six import iteritems
import gensim.downloader as api
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

reload(sys)
sys.setdefaultencoding('utf8')


def get_textlines(filename):
    textlines = [line.strip() for line in open(filename, 'r').readlines()]
    return textlines


def clean_text(sentence, punctuation):
    sentence = re.sub(punctuation.decode('utf8'), "".decode('utf8'),
                      sentence.decode('utf8'))
    return sentence


def cut_words(sentence, stopwords):
    words = list(jieba.cut(sentence))
    words = del_stopwords(words, stopwords)
    return words


def del_stopwords(words, stopwords):
    return [word for word in words if word not in stopwords]


class W2V():

    def __init__(self, words):
        self.words = np.array(words)
        self.dictionary = corpora.Dictionary(words)
        self.corpus = [self.dictionary.doc2bow(word) for word in words]
        self.tfidf = models.TfidfModel(dictionary=self.dictionary)
        # 这里加入字典，字典的id2token就会被初始化
        self.lsi = models.LsiModel(self.tfidf[self.corpus], id2word=self.dictionary, num_topics=300)
        self.similarity = similarities.MatrixSimilarity(self.lsi[self.corpus])
        self.word2vec = Word2Vec(words, min_count=1, window=5, size=200, iter=50)
        words = [word for doc in self.words for word in doc]
        global_corpus = self.get_corpus(words)
        self.init_global_bows(global_corpus)
        self.init_global_tfidfs(global_corpus)

    def init_global_bows(self, global_corpus):
        self.global_bows = {}
        for id, bow in global_corpus:
            token = self.dictionary.id2token[id]
            self.global_bows[token] = bow

    def init_global_tfidfs(self, global_corpus):
        self.global_tfidfs = {}
        for id, tf in global_corpus:
            token = self.dictionary.id2token[id]
            idf = np.log(1.0 * self.tfidf.num_docs / self.tfidf.dfs[id])
            self.global_tfidfs[token] = tf * idf

    def get_vector(self, word):
        return self.word2vec.wv[word]

    def get_id_by_token(self, token):
        return self.dictionary.token2id[token]

    def get_token_by_id(self, id):
        return self.dictionary.id2token[id]

    def get_corpus(self, words=None):
        if words is not None:
            words = np.array(words)
            if words.ndim == 1:
                return self.dictionary.doc2bow(words)
        else:
            words = self.words
        # [[(id1,num1),(id2,num2)],[(id3,num3)]]
        return [self.dictionary.doc2bow(doc) for doc in words]

    def get_bows(self):
        bows = [
            [(self.dictionary.id2token[id], num) for id, num in sorted(doc, key=lambda (id, num): -num)]
            for doc in self.corpus
        ]
        return bows

    def get_tfidf(self, words=None):
        # 在多少文档中出现
        corpus = self.get_corpus(words)
        return [
            [(self.dictionary.id2token[word[0]], word[1]) for word in sorted(doc, key=lambda word: -word[1])]
            for doc in self.tfidf[corpus]
        ]

    def get_global_bow(self, keyword=None, topn=None):
        if keyword is not None:
            return self.global_bows[keyword]
        bows = sorted(self.global_bows.items(), key=lambda (token, bow): bow, reverse=True)
        if topn is not None:
            bows = bows[:topn]
        return bows

    def get_global_tfidf(self, keyword=None, topn=None):
        if keyword is not None:
            return self.global_tfidfs[keyword]
        tfidfs = sorted(self.global_tfidfs.items(), key=lambda (token, tfidf): tfidf, reverse=True)
        if topn is not None:
            tfidfs = tfidfs[:topn]
        return tfidfs

    def get_similarities(self, words):
        corpus = self.get_corpus(words)
        lsi = self.lsi[corpus]
        sims = self.similarity[lsi]
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        return sims

    def draw_vector(self, topn):
        gloabl_tfidf = self.get_global_tfidf(topn=topn)
        vectors = {}
        for token, tfidf in gloabl_tfidf:
            vectors[token] = self.get_vector(token)
            # print tfidf[0], tfidf[1]

        pca = PCA(n_components=2)
        vectors_2d = pca.fit_transform(vectors.values())
        data = []
        for index, key in enumerate(vectors.keys()):
            x, y = vectors_2d[index][0].astype(float), vectors_2d[index][1].astype(float)
            data[index] = [key, x, y]
        # data = np.array(data)
        font = FontProperties(fname='/System/Library/Fonts/PingFang.ttc', size=8)
        for new_vector in data:
            key, x, y = new_vector[0], new_vector[1], new_vector[2]
            plt.text(x, y, key, fontproperties=font)

        # print data.min(axis=0)
        x_min = np.min(np.array(data)[:, 1].astype(float), axis=0) - 0.5
        x_max = np.max(np.array(data)[:, 1].astype(float), axis=0) + 0.5
        y_min = np.min(np.array(data)[:, 2].astype(float), axis=0) - 0.5
        y_max = np.max(np.array(data)[:, 2].astype(float), axis=0) + 0.5
        plt.ylim(y_min, y_max)
        plt.xlim(x_min, x_max)
        plt.show()


def get_words():
    jieba.load_userdict("/Users/a1800101471/PycharmProjects/python_demo2/nlp_demo/acim/userdict_acim.txt")
    raw_textlines = get_textlines('/Users/a1800101471/PycharmProjects/python_demo2/nlp_demo/acim/acim_s.txt')
    stopwords = get_textlines('/Users/a1800101471/PycharmProjects/python_demo2/nlp_demo/acim/stop_word_acim.txt')

    punctuation = "[\t\n\s]" \
                  "|[,\.\!\?\/_]" \
                  "|[､、，：。！？；．.‧…]" \
                  "|[-~`＠＃＄％＾＆＊＋－＝/\\\]" \
                  "|[~@#￥%&*+——／＼〾〿]+" \
                  "|[\"\'〝〞＂‟“”‘’＇｀‛〃〟„]" \
                  "|[﹏—–〰〜＿]" \
                  "|[\(\)\{\}\[\]\<\>＜＞｛｝［］《》〈〉（）「」『』【】〔〕〖〗〘〙〚〛｟｠｢｣]" \
                  "|[⑴⑵⑶⑷⑸⑹⑺⑻]" \
                  "|[a-zA-Z0-9]"
    sentences = [clean_text(sentence, punctuation) for sentence in raw_textlines if
                 (len(sentence) > 0) & (not sentence.isspace())]
    word_matrix = [cut_words(sentence, stopwords) for sentence in sentences]
    return word_matrix


def test1():
    words = get_words()
    w2v = W2V(words)
    print '--------------词  频--------------'
    bows = w2v.get_global_bow()
    for bow in bows[:100]:
        print bow[0], bow[1]
    print '---------------------------------'
    tfidf = w2v.get_tfidf()
    # for tfidf in tfidf[:50]:
    #     for item in tfidf:
    #         print item[0], item[1]
    print '---------------------------------'
    gloabl_tfidf = w2v.get_global_tfidf()
    vectors = {}
    for tfidf in gloabl_tfidf[:100]:
        vectors[tfidf[0]] = w2v.get_vector(tfidf[0])
        print tfidf[0], tfidf[1]

    pca = PCA(n_components=2)
    new_vector = pca.fit_transform(vectors.values())
    new_vectors = []
    for index, key in enumerate(vectors.keys()):
        # new_vectors[key] = new_vector[index]
        new_vectors.append([key, new_vector[index][0], new_vector[index][1]])

    font = FontProperties(fname='/System/Library/Fonts/PingFang.ttc', size=8)
    for new_vector in new_vectors:
        key, x, y = new_vector[0], new_vector[1], new_vector[2]
        plt.text(x, y, key, fontproperties=font)

    x_min = np.min(np.array(new_vectors)[:, 1].astype(float), axis=0) - 0.5
    x_max = np.max(np.array(new_vectors)[:, 1].astype(float), axis=0) + 0.5
    y_min = np.min(np.array(new_vectors)[:, 2].astype(float), axis=0) - 0.5
    y_max = np.max(np.array(new_vectors)[:, 2].astype(float), axis=0) + 0.5
    plt.ylim(y_min, y_max)
    plt.xlim(x_min, x_max)
    plt.show()


if __name__ == '__main__':
    words = get_words()
    w2v = W2V(words)
    print w2v.get_corpus()
    print w2v.get_global_bow(keyword=u'圣灵')
    bows = w2v.get_global_bow(topn=10)
    for token, bow in bows:
        print token, bow
    tfidfs = w2v.get_global_tfidf(topn=10)
    for token, tfidf in tfidfs:
        print token, tfidf
    w2v.draw_vector(10)
