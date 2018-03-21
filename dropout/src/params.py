#!/usr/bin/env python
# -*- coding: utf-8 -*-

# _/_/_/ for data_maker.py

MEAN = 0
STDDEV = 1
SHIFT = 1.7
X_MIN = -7
X_MAX = 7
NOISE_STDDEV = 0.01
SAMPLE_SIZE = 20
XS_PATH = '../dataset/xs.npy'
YS_PATH = '../dataset/ys.npy'

# _/_/_/ for train.py

EPOCHS = 1000
BATCH_SIZE = 10  # SAMPLE_SIZE % BATCH_SIZE == 0
INPUT_SIZE = 1
HIDDEN_SIZE = 512
OUTPUT_SIZE = INPUT_SIZE
LEARNING_RATE = 0.01
DROPOUT_RATIO = 0.05
WEIGHT_DECAY = 0.00001
OUTPUT_DIR_PATH = '../results'
MODEL_NAME = 'model'

# _/_/_/ for calculate_uncertainty.py

LENGTH_SCALE = 10
SQUARED_LENGTH_SCALE = LENGTH_SCALE * LENGTH_SCALE
TAU = SQUARED_LENGTH_SCALE * (1 - DROPOUT_RATIO) / (2 * SAMPLE_SIZE * WEIGHT_DECAY)
SAMPLING_SIZE = 200
ITERATIONS = 100
