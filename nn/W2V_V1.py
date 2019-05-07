#!/usr/bin/env python
# -*- coding: utf-8 -*-
# coding: utf-8
import sys
import numpy as np
from nn.NN_V2 import NN
from nn.Functions import *
#https://ai.yanxishe.com/page/TextTranslation/1317
# https://towardsdatascience.com/word2vec-from-scratch-with-numpy-8786ddd49e72
# https://www.geeksforgeeks.org/implement-your-own-word2vecskip-gram-model-in-python/
#单词到向量的转换也被称为单词嵌入（word embedding）
#每个唯一的单词在空间中被分配一个向量
#Continuous Bag-of-Words(CBOW) 从相邻单词（上下文单词）猜测输出（目标单词) 比skip-gram训练快几倍，对出现频率高的单词的准确度稍微更好一些
#Skip-gram(SG) 从目标单词猜测上下文单词 能够很好地处理少量的训练数据，而且能够很好地表示不常见的单词或短语
#Word2Vec是基于分布假说
#数据清洗 https://www.kdnuggets.com/2018/03/text-data-preprocessing-walkthrough-python.html
#gensim.utils.simple_preprocess，它将文档转换为由小写的词语（Tokens ）组成的列表，并忽略太短或过长的词语
#单词嵌入(word embedding)的维度 100到300，超过300维度会导致效益递减，维度也是隐藏层的大小



def test2():
    sentences = u'一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十'
    # 2个字典
    #将id映射到单词的字典
    id2word = {}
    #单词映射到id的字典
    word2id = {}
    data = list(sentences)
    for index, word in enumerate(set(sentences)):
        print (type(word))
        id = index
        id2word[id] = word
        word2id[word] = id
    vocab_size = len(id2word)

    data_ids = [word2id[word] for word in data]
    #单词进行one-hot编码
    data_onehot = onehot(data_ids)
    x = data_onehot[:-1]
    y = data_onehot[1:]

    nn = NN()
    nn.add(vocab_size, input_dim=vocab_size)
    nn.add(vocab_size, activation='softmax')
    nn.compile()
    nn.train(x, y, 20000, reg_lambda=0)

def test1():

    print("下面我们将讨论一些关于一维数组的乘法的问题")
    A = np.array([1, 2, 3])
    B = np.array([2, 3, 4])
    c = [1, 2, 3]
    print("*:", A * B)  # 对数组执行的是对应位置元素相乘
    print("np.dot():", np.dot(A, B))  # 当dot遇到佚为1，执行按位乘并相加
    print("np.multiply():", np.multiply(A, B))  # 对数组执行的是对应位置的元素相乘
    print("np.outer():", np.outer(A, B))  # A的一个元素和B的元素相乘的到结果的一行

    print("下面我们将讨论一些关于二维数组和二位数组的乘法的问题")
    a = np.array([[1, 2, 3], [3, 4, 5]])
    b = np.array([[1, 1], [2, 2], [3, 3]])
    c = np.array([[2, 2, 2], [3, 3, 3]])
    # 出错：维度不对应：print("*:",a*b)
    print("*:", a * c)  # *对数组执行的是对应位置相乘
    print("np.dot():", np.dot(a, b))  # 当dot遇到佚不为1执行矩阵的乘法（2，3）×（3,2）=（2,2）
    # 出错，维度不对应：print("np.multiply():",np.multiply(a,b))
    print("np.multiply():", np.multiply(a, c))  # 数组或者矩阵对应位置元素相乘，返回的是与原数组或者矩阵的大小一致

    print("下面我们将讨论一些关于矩阵的乘法的问题")
    A = np.mat([[1, 2, 3], [3, 4, 5]])
    B = np.mat([[1, 1], [2, 2], [3, 3]])
    C = np.mat([[2, 2, 2], [3, 3, 3]])
    D = [1, 2, 3]
    print("*:", A * B)  # *对矩阵执行的是矩阵相乘
    print("np.dot():", np.dot(A, B))  # dot对矩阵执行的是矩阵相乘
    print("np.dot():", np.dot(A, D))
    # 这里可以看出矩阵和矩阵的相相乘是轶为2的，所以是执行的矩阵乘法，但是矩阵和向量相乘是轶为1的，执行的是对应相乘加和
    print("np.multiply():", np.multiply(A, C))  # multiply执行的是矩阵对应元素相乘

if __name__ == '__main__':
    test1()