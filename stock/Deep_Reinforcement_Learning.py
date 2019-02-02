# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pandas_datareader.data as web
from subprocess import check_output
import time
import copy
import chainer
import chainer.functions as F
import chainer.links as L
from plotly import tools
from plotly.graph_objs import *
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot, iplot_mpl
#from stock.Environment1 import Environment1
import plotly.graph_objs as go

init_notebook_mode(connected=True)


def plot_train_test(train, test, data_split):
    data = [
        Candlestick(x=train.index, open=train['Open'], high=train['High'], low=train['Low'], close=train['Close'],
                    name='train'),
        Candlestick(x=test.index, open=test['Open'], high=test['High'], low=test['Low'], close=test['Close'],
                    name='test')
    ]
    layout = {
        'shapes': [{'x0': data_split, 'x1': data_split, 'y0': 0, 'y1': 1, 'xref': 'x', 'yref': 'paper'}],
        'annotations': [
            {'x': data_split, 'y': 1.0, 'xref': 'x', 'yref': 'paper', 'showarrow': False, 'xanchor': 'left',
             'text': 'text data'},
            {'x': data_split, 'y': 1.0, 'xref': 'x', 'yref': 'paper', 'showarrow': False, 'xanchor': 'right',
             'text': 'train data'},
        ]
    }
    figure = go.Figure(data=data, layout=layout)
    plot(figure)


data = web.DataReader('GOOG', data_source='yahoo', start='1/1/2008', end='12/30/2018')
data = pd.DataFrame(data)

data_split = '2016-01-01'
train = data[:data_split]
test = data[data_split:]
print len(train), len(test)
# env = Environment1(train)
# print(env.reset())
# for _ in range(3):
#     pact = np.random.randint(3)
#     print(env.step(pact))
plot_train_test(train, test, data_split)