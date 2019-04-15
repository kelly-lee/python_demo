#!/usr/bin/env python
# -*- coding: utf-8 -*-
# coding: utf-8
import sys
import jieba
import re
from collections import Counter
import numpy as np
from gensim.models import word2vec

reload(sys)
sys.setdefaultencoding('utf8')

def get_textlines(filename):
    textlines = [line.strip() for line in open(filename, 'r').readlines()]
    return textlines

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
raw_textlines = get_textlines('acim_s.txt')
stopwords = get_textlines('stop_word_acim.txt')




def clean_text(sentence):
    sentence = re.sub(punctuation.decode('utf8'), "".decode('utf8'),
                      sentence.decode('utf8'))
    return sentence


def cut_words(sentence):
    words = jieba.cut(sentence)
    w1 = words
    # print ", ".join(words)
    # for word in words:

    words = del_stopwords(words)
    print ", ".join(words)

    return words


def del_stopwords(words):
    w=[]
    for word in words:
        if word not in stopwords:
             # print word
             w.append(word)
    # c_w = [word for word in words if word not in stopwords]
    # print 'aaa',c_w
    # print 'bbb', w
    return w



sentences = [clean_text(sentence) for sentence in raw_textlines if (len(sentence) > 0) & (not sentence.isspace())]
word_matrix = [cut_words(sentence) for sentence in sentences]
# words = [word for sent_word in word_matrix for word in sent_word]
# print words[0:2]
# print len(set(words))
# counter = Counter(words)
# for w in counter.most_common(200):
#     print w[0]
model = word2vec.Word2Vec(word_matrix, min_count=5, size=500)
for w in  model.wv.most_similar(u"没有"):
    print w[0],w[1]

# for sentence in  sentences:
#     print sentence
# print len(cleaned_textlines)
# i = 0
# word_matrix = []
#
# for sentence in cleaned_textlines:
#     i = i + 1
#     print np.around(i * 1.0 / len(cleaned_textlines),2)
#     words = cut_words(sentence)
#     # words = del_stopwords(words)
#     word_matrix.append(words)
# word_set = [word for words in word_matrix for word in words]
# print word_set
# print len(word_set)

# words = [del_stopwords(words) for sentence in cleaned_textlines for words in cut_words(sentence)]
# print words

# with open('acim.txt', 'r') as file:
#     sentences = file.readlines()
# # print sentences
# for sentence in sentences:
#     # sentence = sentence.strip()
#
#     if (len(sentence) > 0) & (not sentence.isspace()):
#         print sentence
#         seg_list = jieba.cut(sentence)  # 默认是精确模式
#         print(", ".join(seg_list))
#
# print type(sentences)
#
# temp = "想做/ 兼_职/学生_/ 的 、加,我Q：  1 5.  8 0. ！！？？  8 6 。0.  2。 3     有,惊,喜,哦"
# temp = " ~`＠＃＄％＾＆＊＋－-＝/\         -      -  -  精装-"
# temp = temp.decode("utf8")
# temp = re.sub(punctuation.decode("utf8"), "".decode("utf8"), temp.decode("utf8"))
# print temp
