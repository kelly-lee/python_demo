import pandas as pd
import numpy as np
import jieba
from sklearn.model_selection import train_test_split


# pos = pd.read_excel('pos.xls', header=None, index=None)
# neg = pd.read_excel('neg.xls', header=None, index=None)
# pos['words'] = pos[0].apply(lambda text: list(jieba.cut(text)))
# neg['words'] = neg[0].apply(lambda text: list(jieba.cut(text)))
# y = np.concatenate((np.ones(len(pos)), np.zeros(len(neg))))
# x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos['words'], neg['words'])), y, test_size=0.2)
#
# print len(pos),np.ones(len(pos))
# print len(neg), np.zeros(len(neg))
# print y.shape

string='电池充完了电连手机都打不开.简直烂的要命.真是金玉其外,败絮其中!连5号电池都不如'
words = jieba.lcut(string)
# words_vecs = get_predict_vecs(words)
