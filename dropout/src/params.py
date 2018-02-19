#!/usr/bin/env python
# -*- coding: utf-8 -*-

# _/_/_/ for data_maker.py

MEAN = 0
STDDEV = 1
SHIFT = 1.5
X_MIN = -7
X_MAX = 7
NOISE_STDDEV = 0.03
SAMPLE_SIZE = 20
XS_PATH = '../dataset/xs.npy'
YS_PATH = '../dataset/ys.npy'

# _/_/_/ for train.py

EPOCHS = 10
BATCH_SIZE = 5  # SAMPLE_SIZE % BATCH_SIZE == 0
INPUT_SIZE = 1
HIDDEN_SIZE = 100
OUTPUT_SIZE = INPUT_SIZE
LEARNING_RATE = 1.0e-04
DROPOUT_RATIO = 0.05
WEIGHT_DECAY = 0.00001
SPLIT_RATE = 1.0
OUTPUT_DIR_PATH = '../results'
MODEL_NAME = 'model'
