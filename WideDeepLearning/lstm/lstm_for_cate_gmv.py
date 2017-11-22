#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17/11/17 PM4:45
# @Author  : shaoguang.csg
# @File    : lstm_for_cate_gmv.py

import tensorflow as tf
import pandas as pd
from sklearn import preprocessing, metrics

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

train_file = '/Users/cheng/Data/data/cate_gmv/dm_c2b_v3_cate_deal_train_set_no_nan'
test_file = '/Users/cheng/Data/data/cate_gmv/dm_c2b_v3_cate_deal_test_set_no_nan'





