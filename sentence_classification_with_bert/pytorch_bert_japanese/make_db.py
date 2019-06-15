#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import numpy as np
import bert_juman
import time
from progressbar import ProgressBar
import pickle

HEAD_PATTERN = re.compile(r"^[0-9] ")
PATH = "/home/ubuntu/data/kyodai_bert/disc.txt"
DIR_PATH = "/home/ubuntu/data/kyodai_bert/Japanese_L-12_H-768_A-12_E-30_BPE"
OUT_PATH = "/home/ubuntu/data/kyodai_bert/disc_sentences_reduce_mean_max.npy"
DIM = 768 * 2
SENTENCE_PATH = "/home/ubuntu/data/kyodai_bert/disc_raw_sentences.pkl"


def load_sentences(path):
    sentences = []
    for line in open(path):
        line = line.strip()
        if line is not "":
            m = HEAD_PATTERN.match(line)
            if not m:
                continue
            tokens = line.split()
            assert(len(tokens) == 2)
            sentences.append(tokens[1])
    return sentences


if __name__ == "__main__":
    start = time.time()
    sentences = load_sentences(PATH)
    pickle.dump(sentences, open(SENTENCE_PATH, 'wb'))
    bert = bert_juman.BertWithJumanModel(DIR_PATH, use_cuda=True)
    max = len(sentences)

    print("max {}".format(max))
    m = np.empty((max, DIM))
    p = ProgressBar(0, max)
    for i, sentence in enumerate(sentences[:max]):
        m[i] = bert.get_sentence_embedding(sentence, pooling_strategy="REDUCE_MEAN_MAX")
        p.update(i)
    np.save(OUT_PATH, m)
    end = time.time()
    print("{}[sec]".format(end - start))
