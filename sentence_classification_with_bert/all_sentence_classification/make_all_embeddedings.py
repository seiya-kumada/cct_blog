#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import params
import sys
import numpy as np
sys.path.append("../pytorch_bert_japanese")
import bert_juman  # noqa


ROOT_DIR_PATH = "/home/ubuntu/data/sentence_classification/all_titles/text"
DST_DIR_PATH = "/home/ubuntu/data/sentence_classification/all_titles/sentences/"


def read_document(path):
    lines = []
    for line in open(path):
        line = line.strip()
        if line != "":
            lines.append(line.replace("#", ""))
    lines = lines[3:]
    return lines


def convert_to_embeddings(bert, lines):
    size = len(lines)
    vs = np.empty((size, params.DIM))
    for (i, line) in enumerate(lines):
        vs[i] = bert.get_sentence_embedding(line, pooling_strategy=params.POOLING_STRATEGY)
    return vs


if __name__ == "__main__":
    print("loading model...")
    bert = bert_juman.BertWithJumanModel(params.DIR_PATH, use_cuda=True)
    print("model loading done!")
    for (root, dirs, files) in os.walk(ROOT_DIR_PATH):
        for dir in dirs:
            if dir == "smax" or dir == "sports-watch" or dir == "topic-news" or dir == "livedoor-homme" or dir == "movie-enter":
                continue
            print(dir)
            dst_dir_path = os.path.join(DST_DIR_PATH, dir)
            if not os.path.isdir(dst_dir_path):
                os.makedirs(dst_dir_path)

            dir_path = os.path.join(root, dir)
            for i, line in enumerate(os.listdir(dir_path)):
                if dir in line:
                    path = "/home/ubuntu/data/sentence_classification/all_titles/text/dokujo-tsushin/dokujo-tsushin-5867961.txt"
                    # os.path.join(root, dir, line)
                    lines = read_document(path)
                    print("> [{}]process {}({})".format(i, path, len(lines)))
                    embeddings = convert_to_embeddings(bert, lines)
                    assert(embeddings.shape == (len(lines), params.DIM))
                    dst_path = os.path.join(dst_dir_path, line.replace('txt', 'npy'))
                    np.save(dst_path, embeddings)
