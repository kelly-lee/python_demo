#!/usr/bin/env python
# -*- coding: utf-8 -*-
# coding: utf-8
import sys
import pandas as pd
import numpy as np
from tools import  Preprocessing

train_data = pd.read_csv('data/sales_train.csv',sep=',')
info = Preprocessing.info(train_data)