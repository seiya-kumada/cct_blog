#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import pickle
import sys
import params
from progressbar import ProgressBar
sys.path.append("../pytorch_bert_japanese")
import bert_juman  # noqa
# https://tech.fusic.co.jp/machine-learning/bert-fine-tuning-test/

if __name__ == "__main__":

    bert = bert_juman.BertWithJumanModel(params.DIR_PATH, use_cuda=True)
    sentences = []
    labels = []
    for line in open(params.PATH):
        line = line.strip()
        tokens = line.split('\t')
        label = labels.append(int(tokens[1]))
        sentence = tokens[0]
        sentence = sentence.replace("@", "")
        sentence = sentence.replace("#", "")
        sentences.append(sentence)

    size = len(sentences)
    vs = np.empty((size, params.DIM))
    p = ProgressBar(0, size)
    for i, sentence in enumerate(sentences):
        vs[i] = bert.get_sentence_embedding(sentence, pooling_strategy=params.POOLING_STRATEGY)
        p.update(i)

    head, tail = os.path.split(params.EMBEDDINGS_PATH)
    if not os.path.isdir(head):
        os.mkdir(head)

    np.save(params.EMBEDDINGS_PATH, vs)
    pickle.dump(labels, open(params.LABEL_PATH, 'wb'))
