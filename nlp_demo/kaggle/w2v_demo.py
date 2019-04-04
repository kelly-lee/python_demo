#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import sys
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

import nltk
import nltk.data
from nltk.corpus import stopwords
from sklearn.externals import joblib
from gensim.models.word2vec import Word2Vec


def tokenize(text):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    try:
        text = tokenizer.tokenize(text.strip())
    except:
        print 'error'
    sentences = [clean_text(s) for s in text if s]
    return sentences


def clean_text(text):
    # 去标签
    text = BeautifulSoup(text, 'html.parser').get_text()

    # 去标点
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # 小写
    words = text.lower().split()
    # 去停用词
    stopwords = {}.fromkeys([line.rstrip() for line in open('stopwords.txt')])
    # words = [word for word in words if word not in stopwords]
    return ' '.join(words)


def clean(raw_path, text_path):
    # 清洗句子
    df = pd.read_csv(raw_path, sep='\t', escapechar='\\')
    df['clean_review'] = df.review.apply(clean_text)
    df.to_csv(text_path)


def train(text_path, model_path):
    df = pd.read_csv(text_path)
    vectorizer = CountVectorizer(max_features=5000)
    train_data_features = vectorizer.fit_transform(df['clean_review']).toarray()
    forest = RandomForestClassifier(n_estimators=100)
    forest = forest.fit(train_data_features, df.sentiment)
    matrix = confusion_matrix(df.sentiment, forest.predict(train_data_features))
    print matrix
    joblib.dump(forest, model_path)


def test(test_path, model_path, test_result_path):
    df = pd.read_csv(test_path)
    vectorizer = CountVectorizer(max_features=5000)
    test_data_features = vectorizer.fit_transform(df['clean_review']).toarray()
    model = joblib.load(model_path)
    result = model.predict(test_data_features)
    output = pd.DataFrame({'id': df.id, 'sentiment': result})
    output.to_csv((test_result_path), index=False)


def f():
    clean('labeledTrainData.tsv', 'labeledTrainData.csv')
    clean('unlabeledTrainData.tsv', 'unlabeledTrainData.csv')
    train('labeledTrainData.tsv', 'forest_model.m')
    test('unlabeledTrainData.csv', 'forest_model.m', 'unlabeledTrainData_result.csv')


def f2():
    clean('labeledTrainData.tsv', 'labeledTrainData.csv')
    clean('unlabeledTrainData.tsv', 'unlabeledTrainData.csv')


if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding("utf-8")
    df = pd.read_csv('unlabeledTrainData.tsv', sep='\t', escapechar='\\')
    # df['clean_review'] = df.review.apply(tokenize)
    sentences = sum(df.review.apply(tokenize), [])
    print sentences

    # df = pd.read_csv('unlabeledTrainData.tsv', sep='\t', escapechar='\\')
    # print df.info()
    # tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    # raw_sentences = tokenizer.tokenize(df['review'][0].strip())
    # sentences = [clean_text(s) for s in raw_sentences if s]
    # print raw_sentences
    # print sentences
    # clean('unlabeledTrainData.tsv','unlabeledTrainData_1.tsv')

    # num_features = 300  # Word vector dimensionality
    # min_word_count = 40  # Minimum word count
    # num_workers = 4  # Number of threads to run in parallel
    # context = 10  # Context window size
    # downsampling = 1e-3  # Downsample setting for frequent words
    # print('Training model...')
    # sentences = sum(df.review.apply(split_sentences), [])
    # model = Word2Vec(sentences, workers=num_workers,
    #                           size=num_features, min_count=min_word_count,
    #                           window=context, sample=downsampling)
    # model.init_sims(replace=True)
    # model.save('w2v.m')
