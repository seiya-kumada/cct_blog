#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

DATASET_PATH = './dataset.txt'
M = 1
ALPHA = 0.1
SIGMA = 0.015
TAU = 1 / SIGMA**2
ITER = 70000000
THIN = 3500
BURN = ITER // 2
OUT_DIR_PATH = 'results'
PICKLE_PATH = os.path.join(OUT_DIR_PATH, 'linear_regression_trial-6.pkl')
CSV_PATH = os.path.join(OUT_DIR_PATH, 'linear_regression_trial-6.csv')
XCOUNT = 50
XMIN = 0
XMAX = 4
YMEANS_PATH = os.path.join(OUT_DIR_PATH, 'ymeans.npy')
YSTDS_PATH = os.path.join(OUT_DIR_PATH, 'ystds.npy')
IXS_PATH = os.path.join(OUT_DIR_PATH, 'ixs.npy')
RESULT_PNG_PATH = os.path.join(OUT_DIR_PATH, 'bayes.png')
YPREDICTIONS_PATH = os.path.join(OUT_DIR_PATH, 'ypredictions.npy')
ANSWER_PATH = os.path.join(OUT_DIR_PATH, 'answer_curve.txt')
