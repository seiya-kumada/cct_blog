#!/usr/bin/env python
# -*- coding: utf-8 -*-

POOLING_STRATEGY = "CLS_TOKEN"
DIM = 768
MODE = "cls"
PHASE = "test"
TITLES = "all_titles"
PATH = "/home/ubuntu/data/sentence_classification/{}/{}.txt".format(TITLES, PHASE)
DIR_PATH = "/home/ubuntu/data/kyodai_bert/Japanese_L-12_H-768_A-12_E-30_BPE"
EMBEDDINGS_PATH = "/home/ubuntu/data/sentence_classification/{}/{}/{}_embeddings.npy".format(TITLES, MODE, PHASE)
LABEL_PATH = "/home/ubuntu/data/sentence_classification/{}/{}/{}_labels.pkl".format(TITLES, MODE, PHASE)
LIBSVM_PATH = "/home/ubuntu/data/sentence_classification/{}/{}/{}_libsvm.txt".format(TITLES, MODE, PHASE)
