#!/usr/bin/env python
# -*- coding: utf-8 -*-

# LSTM層の数
N_LAYERS = 1

DROPOUT = 0.5

# fibonacci数列を割る値
VALUE = 5

# 時系列データの全長
TOTAL_SIZE = 10000

# 訓練とテストの分割比
SPRIT_RATE = 0.9

# 入力時の時系列データ長
SEQUENCE_SIZE = 200

EPOCHS = 100
BATCH_SIZE = 100

# 入力層の次元
N_IN = 2

# 隠れ層の次元
N_HIDDEN = 200

# 出力層の次元
N_OUT = 1

GPU = 0

SEED = 0
