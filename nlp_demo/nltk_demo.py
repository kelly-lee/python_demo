# encoding=utf-8
import nltk
import re
from nltk.corpus import brown


emoticons_str = r"""
 (?:
 [:=;] # 眼睛
 [oO\-]? # ⿐⼦
 [D\)\]\(\]/\\OpP] # 嘴
 )"""

regex_str = [
    emoticons_str,
    r'<[^>]+>',  # HTML tags
    r'(?:@[\w_]+)',  # @某⼈
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # 话题标签
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',
    # URLs
    r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # 数字
    r"(?:[a-z][a-z'\-_]+[a-z])",  # 含有 - 和 ‘ 的单词
    r'(?:[\w_]+)',  # 其他
    r'(?:\S)'  # 其他
]




def tokenize(s):
    print s,'a'

    tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)
    return tokens_re.findall(s)


def preprocess(s, lowercase=False):
    emoticon_re = re.compile(r'^' + emoticons_str + '$', re.VERBOSE | re.IGNORECASE)
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in
                  tokens]
    return tokens


from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import SnowballStemmer


# 词干提取
def stem(type='Snowball', word=''):
    if type == 'Porter':
        stemmer = PorterStemmer()
    elif type == 'Lancaster':
        stemmer = LancasterStemmer()
    else:
        stemmer = SnowballStemmer('english')
    return stemmer.stem(word)


from nltk.stem import WordNetLemmatizer


# 词原型提取
# v动词，a形容词，r，n名词
# CC     coordinatingconjunction 并列连词
# CD     cardinaldigit  纯数  基数
# DT     determiner  限定词（置于名词前起限定作用，如 the、some、my 等）
# EX     existentialthere (like:"there is"... think of it like "thereexists")   存在句；存现句
# FW     foreignword  外来语；外来词；外文原词
# IN     preposition/subordinating conjunction介词/从属连词；主从连词；从属连接词
# JJ     adjective    'big'  形容词
# JJR    adjective, comparative 'bigger' （形容词或副词的）比较级形式
# JJS    adjective, superlative 'biggest'  （形容词或副词的）最高级
# LS     listmarker  1)
# MD     modal (could, will) 形态的，形式的 , 语气的；情态的
# NN     noun, singular 'desk' 名词单数形式
# NNS    nounplural  'desks'  名词复数形式
# NNP    propernoun, singular     'Harrison' 专有名词
# NNPS  proper noun, plural 'Americans'  专有名词复数形式
# PDT    predeterminer      'all the kids'  前位限定词
# POS    possessiveending  parent's   属有词  结束语
# PRP    personalpronoun   I, he, she  人称代词
# PRP$  possessive pronoun my, his, hers  物主代词
# RB     adverb very, silently, 副词    非常  静静地
# RBR    adverb,comparative better   （形容词或副词的）比较级形式
# RBS    adverb,superlative best    （形容词或副词的）最高级
# RP     particle     give up 小品词(与动词构成短语动词的副词或介词)
# TO     to    go 'to' the store.
# UH     interjection errrrrrrrm  感叹词；感叹语
# VB     verb, baseform    take   动词
# VBD    verb, pasttense   took   动词   过去时；过去式
# VBG    verb,gerund/present participle taking 动词  动名词/现在分词
# VBN    verb, pastparticiple     taken 动词  过去分词
# VBP    verb,sing. present, non-3d     take 动词  现在
# VBZ    verb, 3rdperson sing. present  takes   动词  第三人称
# WDT    wh-determiner      which 限定词（置于名词前起限定作用，如 the、some、my 等）
# WP     wh-pronoun   who, what 代词（代替名词或名词词组的单词）
# WP$    possessivewh-pronoun     whose  所有格；属有词
# WRB    wh-abverb    where, when 副词

def lemma(words):
    lemmatizer = WordNetLemmatizer()
    post_tags = nltk.pos_tag(words)
    lemma_words = []
    for post_tag in post_tags:
        lemma_word = lemmatizer.lemmatize(post_tag[0])
        print lemma_word
        lemma_words.append(lemma_word)
    return lemma_words


from nltk.corpus import stopwords


# https://www.ranks.nl/stopwords
# 去除停止词
def filter_stopwords(words):
    filtered_words = [word for word in words if word not in stopwords.words('english')]
    return filtered_words


def sentiment_score(words):
    sentiment_dictionary = {}
    for line in open('AFINN/AFINN-111.txt'):
        word, score = line.split('\t')
        sentiment_dictionary[word] = int(score)
    total_score = sum(sentiment_dictionary.get(word, 0) for word in words)
    return total_score


def brown():
    print brown.categories()
    print len(brown.sents())
    print len(brown.words())


def test_tokenize():
    sentence = "hello world"
    tokens = nltk.word_tokenize(sentence)
    print tokens


def test_tokenize_net():
    tweet = 'RT @angelababy: love you baby! :D http://ah.love #168cm'
    print(preprocess(tweet))


def test_stem():
    print stem('Porter', 'maximum'), stem('Lancaster', 'maximum'), stem('Snowball', 'maximum')
    print stem('Porter', 'presumably'), stem('Lancaster', 'presumably'), stem('Snowball', 'presumably')
    print stem('Porter', 'multiply'), stem('Lancaster', 'multiply'), stem('Snowball', 'multiply')
    print stem('Porter', 'provision'), stem('Lancaster', 'provision'), stem('Snowball', 'provision')
    print stem('Porter', 'wenting'), stem('Lancaster', 'wenting'), stem('Snowball', 'wenting')


def test_lemma():
    words = nltk.word_tokenize('what does the fox say')
    print lemma(words)


def test_sentiment_score():
    s = 'this is a happy day'
    words = preprocess(s)
    print sentiment_score(words)


if __name__ == '__main__':
    test_tokenize_net()
    # test_stem()
    # test_lemma()
    test_sentiment_score()