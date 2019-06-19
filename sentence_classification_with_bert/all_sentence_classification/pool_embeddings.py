#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import glob
import numpy as np

ROOT_DIR_PATH = "/home/ubuntu/data/sentence_classification/all_titles/sentences/"
POOLING_STRATEGY = "REDUCE_MAX"
OUT_DIR_PATH = "/home/ubuntu/data/sentence_classification/all_titles/pooling_sentences/max/data"


def convert_to_single_embedding(pooling_strategy, embedding):
    if pooling_strategy == "REDUCE_MEAN":
        return np.mean(embedding, axis=0)
    elif pooling_strategy == "REDUCE_MAX":
        return np.max(embedding, axis=0)
    elif pooling_strategy == "REDUCE_MEAN_MAX":
        return np.r_[np.max(embedding, axis=0), np.mean(embedding, axis=0)]


def convert(dpath, pooling_strategy, out_dir_path):
    names = glob.glob(dpath + "/*.npy")
    for name in names:
        embedding = np.load(name)
        r = convert_to_single_embedding(pooling_strategy, embedding)
        _, tail = os.path.split(name)
        path = os.path.join(out_dir_path, tail)
        np.save(path, r)


if __name__ == "__main__":
    if not os.path.isdir(OUT_DIR_PATH):
        os.makedirs(OUT_DIR_PATH)

    for (root, dirs, files) in os.walk(ROOT_DIR_PATH):
        for dir in dirs:
            print("> {}".format(dir))
            dir_path = os.path.join(root, dir)
            out_dir_path = os.path.join(OUT_DIR_PATH, dir)
            if not os.path.isdir(out_dir_path):
                os.mkdir(out_dir_path)

            convert(dir_path, POOLING_STRATEGY, out_dir_path)
