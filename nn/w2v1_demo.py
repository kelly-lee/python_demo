#!/usr/bin/env python
# -*- coding: utf-8 -*-
# coding: utf-8
import sys
import tensorflow as tf
import numpy as np

contents = u'一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十'
words = list(set(contents))
contentList = list(contents)
wordsMap = {}
count = 0
for word in words:
    wordsMap[word] = count
    count += 1

vocabulary_size = len(words)
embedding_size = 10
batch_size = len(contentList) - 1
print(wordsMap)
print('=========================')
x = np.ndarray(dtype=np.int32, shape=batch_size)
y = np.ndarray(dtype=np.int32, shape=[batch_size])

for i in range(len(contentList) - 1):
    x[i] = wordsMap[contentList[i]]
    y[i] = wordsMap[contentList[i + 1]]

# for i in range(len(contentList) - 1):
#     y[i] = wordsMap[contentList[i]]
#     x[i] = wordsMap[contentList[i+1]]

print(x)
print(x.shape)
print(y)
print(y.shape)
train_inputs = tf.placeholder(tf.int32, shape=[None])
train_labels = tf.placeholder(tf.int32, shape=[None])

embedDic = tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0)
# embedDic = tf.identity([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
embeddings = tf.Variable(embedDic)
embed = tf.nn.embedding_lookup(embeddings, train_inputs)
outLevel = tf.layers.dense(embed, units=vocabulary_size, activation=None)

loss = tf.losses.sparse_softmax_cross_entropy(labels=train_labels, logits=outLevel)

optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
print('==========================')

sess = tf.InteractiveSession()

sess.run(tf.global_variables_initializer())

for i in range(1000):
    sess.run(train_op, feed_dict={train_inputs: x, train_labels: y})
    if i % 100 == 0:
        print('迭代步数： %s, loss: %s' % (i, sess.run(loss, feed_dict={train_inputs: x, train_labels: y})))


# print(sess.run(embedDic))


def getNextWord(w):
    print('==================================================')
    print('计算 %s 的下一个单词' % w)
    i1 = wordsMap[w.decode('utf8')]
    outV = sess.run(outLevel, feed_dict={train_inputs: [i1]})
    print(outV)
    i2 = sess.run(tf.argmax(outV, 1))[0]
    print('单词索引： %s' % i2)
    print("下一个单词：%s" % words[i2].encode('utf8'))


getNextWord('一')
getNextWord('二')
getNextWord('三')
getNextWord('四')
getNextWord('五')
getNextWord('六')
getNextWord('七')
getNextWord('八')
getNextWord('九')
getNextWord('十')


def distance(w1, w2):
    print('===================================================')
    print('计算单词之间的相似度, word1: %s, word2: %s' % (w1, w2))
    i1 = wordsMap[w1.decode('utf8')]
    i2 = wordsMap[w2.decode('utf8')]
    v1 = tf.nn.embedding_lookup(embeddings, i1)
    # print('单词 %s 的词向量 ： %s' % (w1, sess.run(v1)))
    v2 = tf.nn.embedding_lookup(embeddings, i2)
    # print('单词 %s 的词向量 ： %s' % (w2, sess.run(v2)))
    dis = sess.run(tf.sqrt(tf.reduce_sum(tf.square(v1 - v2))))
    print("单词 word1: %s，word2: %s 之间的距离： %s" % (w1, w2, dis))


distance('五', '一')
distance('五', '二')
distance('一', '二')
distance('一', '三')
distance('一', '四')
distance('一', '五')
distance('一', '六')
distance('一', '七')
distance('一', '八')
distance('一', '九')
distance('一', '十')


def showDis():
    print('=====================================================')
    for word in list(u'一二三四五六七八九十'):
        id = wordsMap[word.decode('utf8')]
        print('word: %s, 向量：%s' % (word, sess.run(tf.nn.embedding_lookup(embeddings, id))))


showDis()
