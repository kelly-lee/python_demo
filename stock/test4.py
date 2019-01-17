# -*- coding: utf-8 -*-

# from __future__ import print_function

# Author: Gael Varoquaux gael.varoquaux@normalesup.org
# License: BSD 3 clause
import tushare as ts
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import TushareStore as store
import pandas as pd
import pandas_datareader.data as web
from sklearn import cluster, covariance, manifold

from matplotlib.font_manager import FontProperties

print(__doc__)

pf = pd.read_csv('NASDAQ_companylist.csv')

