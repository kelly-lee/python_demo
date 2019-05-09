# -*- coding: utf-8 -*-
from gensim.models import word2vec
from gensim import corpora, models, similarities
import logging
import numpy as np
from six import iteritems


def test1():
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=logging.INFO)
    print logging.info("begin %s" % 'hello')

    raw_sentences = ['the quick brown fox jumps over the lazy dogs', 'yoyoyo you go home now to sleep']
    words = [raw_sentence.split() for raw_sentence in raw_sentences]
    print words
    num_features = 300  # Word vector dimensionality
    min_word_count = 40  # Minimum word count
    num_workers = 4  # Number of threads to run in parallel
    context = 10  # Context window size
    downsampling = 1e-3  # Downsample setting for frequent words

    # sentences=None, \
    #           corpus_file=None, \
    #
    #                            alpha=0.025, \
    #
    #
    #                  max_vocab_size=None， seed=1,  min_alpha=0.0001,
    #                  sg=0,  ns_exponent=0.75, cbow_mean=1, hashfxn=hash,  null_word=0,
    #                  trim_rule=None, sorted_vocab=1, batch_words=MAX_WORDS_IN_BATCH, compute_loss=False, callbacks=(),
    #                  max_final_vocab=Non
    # min_count 忽略出现min_count以下次数的单词，默认5
    # window 上下文窗口大小，默认5
    # size 词向量的维度，默认100
    # workers 线程数，默认3
    # negative 负采样数量，默认5
    # hs 使用层次，对罕见字有力，默认0
    # sample 词频采样阀值，高于此阀值会欠采样，默认 1e-3，范围 0，1e-5
    # iter 迭代次数 默认5

    model = word2vec.Word2Vec(words, min_count=1)
    print model.wv.similarity('dogs', 'you')


def test2():
    pass


class W2V():
    def __init__(self, words):
        self.words = np.array(words)
        self.dictionary = corpora.Dictionary(words)
        self.token2id = self.dictionary.token2id
        self.id2token = dict(zip(self.token2id.values(), self.token2id.keys()))
        self.corpus = [self.dictionary.doc2bow(word) for word in words]
        self.tfidf = models.TfidfModel(self.corpus)
        self.lsi = models.LsiModel(self.tfidf[self.corpus], id2word=self.dictionary, num_topics=300)
        self.similarity = similarities.MatrixSimilarity(self.lsi[self.corpus])

    def get_id_by_token(self, token):
        return self.token2id[token]

    def get_token_by_id(self, id):
        return self.id2token[id]

    def get_bow(self):
        return [
            [(self.id2token[word[0]], word[1]) for word in sorted(doc, key=lambda word: -word[1])]
            for doc in self.corpus
        ]

    def get_global_bow(self):
        global_corpus = [word for doc in self.corpus for word in doc]
        return [(self.id2token[word[0]], word[1]) for word in sorted(global_corpus, key=lambda word: -word[1])]




# return sorted(enumerate(doc2bow), key=lambda item: -item[1])


def test1():
    # words 2D数组 [[w1,w2],[w3]]-> Dictionary
    # [w1,w2]-> dictionary.doc2bow([w1,w2]) ->corpus -> [(w1_id,w1_count),(w2_id,w2_count)]
    # corpus [[(0,1),(1,2)],[3,5]] -> models.TfidfModel(corpus) -> tfidf_model ->[(w1_id,w1_tfidf),(w2_id,w2_tfidf)]
    # tfidf_model
    documents = ["Human  Human machine interface for lab abc computer applications",
                 "A survey of user opinion of computer system response time",
                 "The EPS user interface management system",
                 "System and human system engineering testing of EPS",
                 "Relation of user perceived response time to error measurement",
                 "The generation of random binary unordered trees",
                 "The intersection graph of paths in trees",
                 "Graph minors IV Widths of trees and well quasi ordering",
                 "Graph minors A survey"]
    documents = ["我 爱 北京 天安门",
                 "天安门 上 太阳 升",
                 "伟大 领袖 毛主席",
                 "指引 我们 向前 进"]
    stopwords = set('for a of the and to in'.split())
    words = [[word for word in document.lower().split() if word not in stopwords] for document in documents]
    print words
    print '-------------------------------------------------------------------------------------------------------'
    # 创建字典，为每个词分配id
    dictionary = corpora.Dictionary(words)
    token2id = dictionary.token2id
    id2token = dict(zip(token2id.values(), token2id.keys()))
    print dictionary
    print token2id
    print id2token
    print '-------------------------------------------------------------------------------------------------------'

    # 统计词频
    # 按每个句子／文档统计
    corpus = [dictionary.doc2bow(word) for word in words]
    print('corpus', corpus)
    # 按所有句子／文档统计
    word_list = np.array([word for word_list in words for word in word_list]).flatten()
    bows = dictionary.doc2bow(word_list)
    print bows
    doc2bow = [(id2token[bow[0]], bow[1]) for bow in bows]
    print doc2bow
    # 另一种获得词频的方法
    for tokenid, docfreq in iteritems(dictionary.dfs):
        print tokenid, id2token[tokenid], docfreq
    print '-------------------------------------------------------------------------------------------------------'
    # 统计 tf-idf
    tfidf_model = models.TfidfModel(corpus)
    corpus_tfidf = tfidf_model[corpus]
    for doc_tfidf in corpus_tfidf:
        print [(id2token[word_tfidf[0]], word_tfidf[1]) for word_tfidf in doc_tfidf]
    print '-------------------------------------------------------------------------------------------------------'
    # 主题推断
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)
    corpus_lsi = lsi[corpus_tfidf]
    print lsi.print_topics(2)
    for doc_lsi in corpus_lsi:
        print doc_lsi
    print '-------------------------------------------------------------------------------------------------------'
    index = similarities.MatrixSimilarity(lsi[corpus])

    doc = "我们"
    vec_bow = dictionary.doc2bow(doc.lower().split())
    vec_lsi = lsi[vec_bow]
    sims = index[vec_lsi]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    print(sims)
    print '-------------------------------------------------------------------------------------------------------'


if __name__ == '__main__':
    documents = ["Human  Human machine interface for lab abc computer applications",
                 "A survey of user opinion of computer system response time",
                 "The EPS user interface management system",
                 "System and human system engineering testing of EPS",
                 "Relation of user perceived response time to error measurement",
                 "The generation of random binary unordered trees",
                 "The intersection graph of paths in trees",
                 "Graph minors IV Widths of trees and well quasi ordering",
                 "Graph minors A survey"]
    stopwords = set('for a of the and to in'.split())
    words = [[word for word in document.lower().split() if word not in stopwords] for document in documents]
    w2v = W2V(words)
    print w2v.get_id_by_token('testing')
    print w2v.get_token_by_id(1)
    bows = w2v.get_bow()
    for bow in bows:
        print bow
    bows = w2v.get_global_bow()
    for bow in bows:
        print bow
