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


def clean_text(sentence):
    sentence = re.sub(punctuation.decode('utf8'), "".decode('utf8'),
                      sentence.decode('utf8'))
    return sentence


def cut_words(sentence):
    words = list(jieba.cut(sentence))
    words = del_stopwords(words)
    # print ", ".join(words)
    return words


def del_stopwords(words):
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

    def get_vector(self, word):
        return self.word2vec.wv[word]

    def get_id_by_token(self, token):
        return self.dictionary.token2id[token]

    def get_token_by_id(self, id):
        return self.dictionary.id2token[id]

    def get_corpus(self, words=None):
        if words is not None:
            words = np.array(words)
            # # if words.dmin == 1:
            # words = words.reshape(-1, 1)
        else:
            words = self.words
        return [self.dictionary.doc2bow(word) for word in words]

    def get_bow(self, keyword=None, topn=None):
        bows = [
            [(self.id2token[word[0]], word[1]) for word in sorted(doc, key=lambda word: -word[1])]
            for doc in self.corpus
        ]
        if keyword is not None:
            bows = bows[keyword]
        if topn is not None:
            bows = bows[:topn]
        return bows

    def get_global_bow(self):
        words = [[word for doc in self.words for word in doc]]
        global_corpus = self.get_corpus(words)
        return [(self.dictionary.id2token[word[0]], word[1]) for word in
                sorted(global_corpus[0], key=lambda word: -word[1])]

    def get_tfidf(self, words=None):
        # 在多少文档中出现
        corpus = self.get_corpus(words)
        return [
            [(self.dictionary.id2token[word[0]], word[1]) for word in sorted(doc, key=lambda word: -word[1])]
            for doc in self.tfidf[corpus]
        ]

    def get_global_tfidf(self):
        idfs = self.tfidf.idfs
        words = [[word for doc in self.words for word in doc]]
        global_tf = self.get_corpus(words)[0]
        global_tfidf = {}
        for key, tf in global_tf:
            idf = np.log(1.0 * self.tfidf.num_docs / self.tfidf.dfs[key])
            token = self.dictionary.id2token[key]
            global_tfidf[token] = tf * idf
        global_tfidf = sorted(global_tfidf.items(), key=lambda word: word[1], reverse=True)
        return global_tfidf

    def get_similarities(self, words):
        corpus = self.get_corpus(words)
        lsi = self.lsi[corpus]
        sims = self.similarity[lsi]
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        return sims


# acim_raw = open('acim.txt').read()
# print acim_raw
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
jieba.load_userdict("/Users/a1800101471/PycharmProjects/python_demo2/nlp_demo/acim/userdict_acim.txt")
raw_textlines = get_textlines('/Users/a1800101471/PycharmProjects/python_demo2/nlp_demo/acim/acim.txt')
stopwords = get_textlines('/Users/a1800101471/PycharmProjects/python_demo2/nlp_demo/acim/stop_word_acim.txt')

sentences = [clean_text(sentence) for sentence in raw_textlines if (len(sentence) > 0) & (not sentence.isspace())]
word_matrix = [cut_words(sentence) for sentence in sentences]

# model = word2vec.Word2Vec(word_matrix, min_count=5, size=500)
# for w in model.wv.most_similar(u"没有"):
#     print w[0], w[1]

w2v = W2V(word_matrix)

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

x_min = np.min(np.array(new_vectors)[:, 1].astype(float), axis=0)-0.5
x_max = np.max(np.array(new_vectors)[:, 1].astype(float), axis=0)+0.5
y_min = np.min(np.array(new_vectors)[:, 2].astype(float), axis=0)-0.5
y_max = np.max(np.array(new_vectors)[:, 2].astype(float), axis=0)+0.5
plt.ylim(y_min, y_max)
plt.xlim(x_min, x_max)
plt.show()
