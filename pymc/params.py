#!/usr/bin/env python
# -*- coding: utf-8 -*-

DATASET_PATH = './dataset.txt'
M = 5
ALPHA = 0.1
SIGMA = 0.015
TAU = 1 / SIGMA**2
# ITER = 20000000
ITER = 10000000
BURN = ITER // 2
THIN = 10
PICKLE_NAME = 'linear_regression.pkl'
XCOUNT = 50
XMIN = -1
XMAX = 4.5
