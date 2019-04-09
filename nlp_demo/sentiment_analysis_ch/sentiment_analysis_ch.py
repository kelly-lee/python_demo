#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import jieba
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from gensim.models.word2vec import Word2Vec
from sklearn.svm import SVC


def do_train_test_split():
    pos = pd.read_excel('pos.xls', header=None, index=None)
    neg = pd.read_excel('neg.xls', header=None, index=None)
    pos['words'] = pos[0].apply(lambda text: list(jieba.cut(text)))
    neg['words'] = neg[0].apply(lambda text: list(jieba.cut(text)))
    x = np.concatenate((pos['words'], neg['words']))
    y = np.concatenate((np.ones(len(pos)), np.zeros(len(neg))))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    np.save('sentiment_analysis_ch_x_train.npy', x_train)
    np.save('sentiment_analysis_ch_y_train.npy', y_train)
    np.save('sentiment_analysis_ch_x_test.npy', x_test)
    np.save('sentiment_analysis_ch_y_test.npy', y_test)
    return x_train, x_test, y_train, y_test


def w2v_train():
    n_dim = 300
    x_train = np.load('sentiment_analysis_ch_x_train.npy')
    x_test = np.load('sentiment_analysis_ch_x_test.npy')
    # 初始化模型和词表
    w2v = Word2Vec(size=n_dim, min_count=10)
    # 单词到整数映射
    w2v.build_vocab(x_train)
    # 在评论训练集上建模(可能会花费几分钟)
    w2v.train(x_train, epochs=w2v.epochs, total_examples=w2v.corpus_count)


    sentence_vector = [build_sentence_vector(z, n_dim, w2v) for z in x_train]
    train_vecs = np.concatenate(sentence_vector)
    np.save('sentiment_analysis_ch_x_train_vecs.npy', train_vecs)
    print train_vecs.shape
    # 在测试集上训练
    w2v.train(x_test, epochs=w2v.epochs, total_examples=w2v.corpus_count)
    w2v.save('sentiment_analysis_ch_w2v.pkl')
    # Build test tweet vectors then scale
    sentence_vector = [build_sentence_vector(z, n_dim, w2v) for z in x_test]
    test_vecs = np.concatenate(sentence_vector)
    # test_vecs = scale(test_vecs)
    np.save('sentiment_analysis_ch_x_test_vecs.npy', test_vecs)

def svm_train():
    x_train_vecs = np.load('sentiment_analysis_ch_x_train_vecs.npy')
    y_train = np.load('sentiment_analysis_ch_y_train.npy')
    x_test_vecs = np.load('sentiment_analysis_ch_x_test_vecs.npy')
    y_test = np.load('sentiment_analysis_ch_y_test.npy')
    clf = SVC(kernel='rbf', verbose=True)
    clf.fit(x_train_vecs, y_train)
    joblib.dump(clf, 'sentiment_analysis_ch_svm.pkl')
    print clf.score(x_test_vecs, y_test)

#
# print len(pos),np.ones(len(pos))
# print len(neg), np.zeros(len(neg))
# print y.shape

def build_sentence_vector(text, size, imdb_w2v):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += imdb_w2v[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    print text, vec
    return vec


def svm_predict(string):
    words = jieba.lcut(string)
    n_dim = 300
    imdb_w2v = Word2Vec.load('sentiment_analysis_ch_w2v.pkl')
    words_vecs = build_sentence_vector(words, n_dim, imdb_w2v)
    clf = joblib.load('sentiment_analysis_ch_svm.pkl')

    result = clf.predict(words_vecs)

    if int(result[0]) == 1:
        print string, ' positive'
    else:
        print string, ' negative'

if __name__ == '__main__' :
    # do_train_test_split()
    # w2v_train()
    # svm_train()
    string='电池充完了电连手机都打不开.简直烂的要命.真是金玉其外,败絮其中!连5号电池都不如'
    svm_predict(string)
