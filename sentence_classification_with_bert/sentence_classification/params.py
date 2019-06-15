#!/usr/bin/env python
# -*- coding: utf-8 -*-

POOLING_STRATEGY = "CLS_TOKEN"
DIM = 768
MODE = "cls"
PHASE = "train"
PATH = "/home/ubuntu/data/sentence_classification/3_titles/{}.txt".format(PHASE)
DIR_PATH = "/home/ubuntu/data/kyodai_bert/Japanese_L-12_H-768_A-12_E-30_BPE"
EMBEDDINGS_PATH = "/home/ubuntu/data/sentence_classification/3_titles/{}/{}_embeddings.npy".format(MODE, PHASE)
LABEL_PATH = "/home/ubuntu/data/sentence_classification/3_titles/{}/{}_labels.pkl".format(MODE, PHASE)
LIBSVM_PATH = "/home/ubuntu/data/sentence_classification/3_titles/{}/{}_libsvm.txt".format(MODE, PHASE)
