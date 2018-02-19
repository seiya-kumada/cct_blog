#!/usr/bin/env python
# -*- coding: utf-8 -*-

# _/_/_/ for data_maker.py

MEAN = 0
STDDEV = 1
SHIFT = 1.5
X_MIN = -7
X_MAX = 7

SAMPLE_SIZE = 1000
INPUT_DIM = 10
XS_PATH = '../dataset/xs.npy'
YS_PATH = '../dataset/ys.npy'

# _/_/_/ for train.py

EPOCHS = 1000
BATCH_SIZE = 10
INPUT_SIZE = INPUT_DIM
HIDDEN_SIZE = 1000
OUTPUT_SIZE = INPUT_DIM
LEARNING_RATE = 1.0e-04
DROPOUT_RATIO = 0.2
WEIGHT_DECAY = 0.0001
SPLIT_RATE = 0.9
OUTPUT_DIR_PATH = '../results'
MODEL_NAME = 'model'
