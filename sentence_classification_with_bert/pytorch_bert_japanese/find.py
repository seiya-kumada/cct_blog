#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import numpy as np
import bert_juman
import pickle
from annoy import AnnoyIndex
# https://qiita.com/wasnot/items/20c4f30a529ae3ed5f52
DB_PATH = "/home/ubuntu/data/kyodai_bert/disc_sentences_reduce_max.npy"
DIR_PATH = "/home/ubuntu/data/kyodai_bert/Japanese_L-12_H-768_A-12_E-30_BPE"
SENTENCE_PATH = "/home/ubuntu/data/kyodai_bert/disc_raw_sentences.pkl"
POOLING_STRATEGY = "REDUCE_MAX"

if __name__ == "__main__":
    db = np.load(DB_PATH)
    sentences = pickle.load(open(SENTENCE_PATH, 'rb'))

    print("db loading done")

    t = AnnoyIndex(db.shape[1])
    for i, v in enumerate(db):
        t.add_item(i, v)
    t.build(10)  # what's it?

    bert = bert_juman.BertWithJumanModel(DIR_PATH, use_cuda=True)
    top = 10
    for line in sys.stdin:
        line = line.strip()
        v = bert.get_sentence_embedding(line, pooling_strategy=POOLING_STRATEGY)
        r, d = t.get_nns_by_vector(v, top, include_distances=True)
        for i in range(top):
            print("[{}]{}({})".format(i, sentences[r[i]], d[i]))
