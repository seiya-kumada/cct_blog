#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
import random
RATE = 0.7

ROOT_DIR_PATH = "/home/ubuntu/data/sentence_classification/all_titles/pooling_sentences/max/data"
LABEL_MAP = {
    "it-life-hack": 0,
    "movie-enter": 1,
    "sports-watch": 2,
    "dokujo-tsushin": 3,
    "kaden-channel": 4,
    "livedoor-homme": 5,
    "peachy": 6,
    "smax": 7,
    "topic-news": 8,
}


def covert_to_libsvm(label, embedding):
    line = str(label)
    for i, v in enumerate(embedding, start=1):
        if abs(v) < 1.0e-6:
            continue
        line += " {}:{}".format(i, v)
    return line


def make_libsvm(dir_path, label, lines):
    names = glob.glob(dir_path + "/*.npy")
    for name in names:
        embedding = np.load(name)
        line = covert_to_libsvm(label, embedding)
        lines.append(line)


def save(lines, start, end, name):
    path = os.path.join(ROOT_DIR_PATH, "{}_libsvm.txt".format(name))
    with open(path, 'w') as fout:
        fout.write("\n".join(lines[start:end]))


if __name__ == "__main__":
    lines = []
    for (root, dirs, files) in os.walk(ROOT_DIR_PATH):
        for dir in dirs:
            print("> {}".format(dir))
            dir_path = os.path.join(root, dir)
            make_libsvm(dir_path, LABEL_MAP[dir], lines)
    random.shuffle(lines)

    total_size = len(lines)
    train_size = int(RATE * total_size)
    test_size = total_size - train_size
    save(lines, 0, train_size, "train")
    save(lines, train_size, total_size, "test")
