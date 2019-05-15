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
import pandas as pd
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
        self.word2vec = Word2Vec(words, min_count=1, window=5, size=200, iter=50, )
        words = [word for doc in self.words for word in doc]
        global_corpus = self.get_corpus(words)
        self.init_global_bows(global_corpus)
        self.init_global_tfidfs(global_corpus)

    def get_word2vec(self):
        return self.word2vec

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

    def get_2d_vectors(self, topn):
        df = pd.DataFrame()
        gloabl_tfidf = self.get_global_tfidf(topn=topn)
        vectors = []
        for token, tfidf in gloabl_tfidf:
            vector = self.get_vector(token)
            vectors.append(vector)
            df = df.append(pd.DataFrame([[vector]], index=[token], columns=['vector']))
        pca = PCA(n_components=2, random_state=0)
        vectors_2d = pca.fit_transform(vectors)
        df['vector_x'] = vectors_2d[:, 0]
        df['vector_y'] = vectors_2d[:, 1]
        return df

    def draw_vector(self, topn):
        data = self.get_2d_vectors(topn)
        font = FontProperties(fname='/System/Library/Fonts/PingFang.ttc', size=8)
        for index, row in data.iterrows():
            plt.text(row['vector_x'], row['vector_y'], index, fontproperties=font)
        desc = data.describe()
        x_min = desc.at['min', 'vector_x'] * 0.8
        x_max = desc.at['max', 'vector_x'] * 1.2
        y_min = desc.at['min', 'vector_y'] * 0.8
        y_max = desc.at['max', 'vector_y'] * 1.2
        plt.ylim(y_min, y_max)
        plt.xlim(x_min, x_max)
        plt.show()

    def get_similarities(self, words):
        corpus = self.get_corpus(words)
        lsi = self.lsi[corpus]
        sims = self.similarity[lsi]
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        return sims


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
    # print w2v.get_corpus()
    # print w2v.get_global_bow(keyword=u'圣灵')
    bows = w2v.get_global_bow(topn=100)
    for token, bow in bows:
        print token, bow
    tfidfs = w2v.get_global_tfidf(topn=100)
    for token, tfidf in tfidfs:
        print token, tfidf
    w2v.draw_vector(100)
    word2vec = w2v.get_word2vec()
    most_similar_cosmuls = word2vec.wv.most_similar_cosmul(positive=[u'爱', u'正念'], negative=[u'恐惧'])
    for token, most_similar_cosmul in most_similar_cosmuls:
        print token, most_similar_cosmul
    print(word2vec.wv.doesnt_match(u'奇迹 上主 救赎 恐惧 心灵 时间 错误'.split()))
    print word2vec.wv.similar_by_word(u'')

    # >> >
    # >> > similarity = word_vectors.similarity('woman', 'man')
    # >> > similarity > 0.8
    # True
    # >> >
    # >> > result = word_vectors.similar_by_word("cat")
    # >> > print("{}: {:.4f}".format(*result[0]))
    # dog: 0.8798
    # >> >
    # >> > sentence_obama = 'Obama speaks to the media in Illinois'.lower().split()
    # >> > sentence_president = 'The president greets the press in Chicago'.lower().split()
    # >> >
    # >> > similarity = word_vectors.wmdistance(sentence_obama, sentence_president)
    # >> > print("{:.4f}".format(similarity))
    # 3.4893
    # >> >
    # >> > distance = word_vectors.distance("media", "media")
    # >> > print("{:.1f}".format(distance))
    # 0.0
    # >> >
    # >> > sim = word_vectors.n_similarity(['sushi', 'shop'], ['japanese', 'restaurant'])
    # >> > print("{:.4f}".format(sim))
    # 0.7067
    # >> >
    # >> > vector = word_vectors['computer']  # numpy vector of a word
    # >> > vector.shape
    # (100,)


if __name__ == '__main__':
    test1()
