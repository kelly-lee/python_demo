# -*- coding: utf-8 -*-
from gensim.models import word2vec
import logging

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
