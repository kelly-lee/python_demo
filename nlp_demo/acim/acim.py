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
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from mittens import GloVe


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
    print sentence
    print ','.join(words)
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
        # size 200~300最佳
        # window 8 最佳
        self.word2vec = Word2Vec(words, min_count=3, window=5, size=200, iter=10)
        words = [word for doc in self.words for word in doc]
        global_corpus = self.get_corpus(words)
        self.init_global_bows(global_corpus)
        self.init_global_tfidfs(global_corpus)
        self.init_cooccurrence_matrix()
        glove_model = GloVe(n=200, max_iter=500)
        self.glove = glove_model.fit(self.cooccurrence_matrix)

    def init_cooccurrence_matrix(self):
        size = len(self.dictionary.token2id)
        matrix = np.zeros((size, size))
        windows = 8
        for doc_words in self.words:
            for center_index, center_word in enumerate(doc_words):
                center_id = self.dictionary.token2id[center_word]
                context_start_index = max(center_index - windows, 0)
                context_end_index = min(center_index + windows + 1, len(doc_words))
                # window_words = doc_words[context_start_index:context_end_index]
                for context_index in range(context_start_index, context_end_index):
                    if center_index != context_index:
                        context_word = doc_words[context_index]
                        context_id = self.dictionary.token2id[context_word]
                        matrix[center_id, context_id] += 1
        self.cooccurrence_matrix = matrix

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

    def get_glove_vector(self, word):
        return self.glove[self.dictionary.token2id[word]]

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
            vector = self.get_glove_vector(token)
            vectors.append(vector)
            df = df.append(pd.DataFrame([[vector]], index=[token], columns=['vector']))
        pca = PCA(n_components=2, random_state=0)
        vectors_2d = pca.fit_transform(vectors)
        df['vector_x'] = vectors_2d[:, 0]
        df['vector_y'] = vectors_2d[:, 1]
        return df

    def draw_vector(self, data):
        font = FontProperties(fname='/System/Library/Fonts/PingFang.ttc', size=8)
        for index, row in data.iterrows():
            color = 'red' if index in [u'关系'] else 'black'
            print row['vector_x'], row['vector_y'], index
            plt.text(row['vector_x'], row['vector_y'], index, fontproperties=font, color=color)
        desc = data.describe()
        print desc
        x_min = desc.at['min', 'vector_x'] * 1
        x_max = desc.at['max', 'vector_x'] * 1
        y_min = desc.at['min', 'vector_y'] * 1
        y_max = desc.at['max', 'vector_y'] * 1
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
    raw_textlines = get_textlines('/Users/a1800101471/PycharmProjects/python_demo2/nlp_demo/acim/acim_8.txt')
    stopwords = get_textlines('/Users/a1800101471/PycharmProjects/python_demo2/nlp_demo/acim/stop_word_acim.txt')

    punctuation = "[\t\n\s　]" \
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
    topn = 50
    bows = w2v.get_global_bow(topn=topn)
    for token, bow in bows:
        print token, bow
    print '----------------------------------'
    tfidfs = w2v.get_global_tfidf(topn=topn)
    for token, tfidf in tfidfs:
        print token
    w2v.draw_vector(topn)
    word2vec = w2v.get_word2vec()
    # most_similar_cosmuls = word2vec.wv.most_similar_cosmul(positive=[u'爱', u'正念'], negative=[u'恐惧'])
    # for token, most_similar_cosmul in most_similar_cosmuls:
    #     print token, most_similar_cosmul
    # print(word2vec.wv.doesnt_match(u'奇迹 上主 救赎 恐惧 心灵 时间 错误'.split()))
    for token, similar in word2vec.wv.similar_by_word(u'关系'):
        print token, similar
    print '----------------------'


def test2():
    words = get_words()
    w2v = W2V(words)
    data = w2v.get_2d_vectors(50)
    w2v.draw_vector(data)


# similar_by_word("cat")
# similarity('woman', 'man')
# wmdistance(sentence_obama, sentence_president)
# word_vectors.distance("media", "media"
# word_vectors.n_similarity(['sushi', 'shop'], ['japanese', 'restaurant'])


if __name__ == '__main__':
    test2()
