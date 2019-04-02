#!/usr/bin/env python
# -*- coding: utf-8 -*-
# process_wiki_data.py 用于解析XML，将XML的wiki数据转换为text格式

import logging
import os.path
import sys
import multiprocessing
import jieba
import jieba.analyse
import jieba.posseg as pseg  # 引入词性标注接口
import codecs, sys
import gensim
from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


def process_wiki_data(xml_path, text_path):
    program = os.path.basename('w2v_demo.py')
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
    # check and process input arguments
    inp, outp = xml_path, text_path
    space = " "
    i = 0
    output = open(outp, 'w')
    wiki = WikiCorpus(inp, lemmatize=False, dictionary={})
    for text in wiki.get_texts():
        output.write(space.join(text) + "\n")
        i = i + 1
        if (i % 10000 == 0):
            logger.info("Saved " + str(i) + " articles")
    output.close()
    logger.info("Finished Saved " + str(i) + " articles")


def seg_by_jieba(text_path, seg_path):
    f = codecs.open(text_path, 'r', encoding='utf-8')
    target = codecs.open(seg_path, 'w', encoding='utf-8')
    print('open files.')

    lineNum = 1
    line = f.readline()
    while line:
        print('---processing ', lineNum, ' article---')
        seg_list = jieba.cut(line, cut_all=False)
        line_seg = ' '.join(seg_list)
        target.writelines(line_seg)
        lineNum = lineNum + 1
        line = f.readline()

    print('well done.')
    f.close()


def train_word2vec_model(seg_path, model_path, vector_path):
    program = os.path.basename('w2v_demo.py')
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
    # check and process input arguments
    inp, outp1, outp2 = seg_path, model_path, vector_path
    model = Word2Vec(LineSentence(inp), size=400, window=5, min_count=5,
                     workers=multiprocessing.cpu_count())
    # trim unneeded model memory = use(much) less RAM
    # model.init_sims(replace=True)
    model.save(outp1)
    model.wv.save_word2vec_format(outp2, binary=False)

def test():
    model = gensim.models.Word2Vec.load("wiki.zh.text.model")
    result =  model.most_similar(u"足球")
    for e in result:
        print e[0],e[1]

if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding("utf-8")
    # process_wiki_data('zhwiki-latest-pages-articles4.xml-p2654618p2771086.bz2', 'wiki.zh.text')
    # seg_by_jieba('wiki.zh.text', 'wiki.zh.text.seg')
    # train_word2vec_model('wiki.zh.text.seg', 'wiki.zh.text.model', 'wiki.zh.text.vector')
    test()
