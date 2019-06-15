#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import random

ROOT_DIR_PATH = "/home/ubuntu/data/sentence_classification/3_text/"
LABEL_MAP = {
    "dokujo-tsushin": 0,
    "it-life-hack": 1,  #
    "kaden-channel": 2,
    "livedoor-homme": 3,
    "movie-enter": 4,  #
    "peachy": 5,
    "smax": 6,
    "sports-watch": 7,  #
    "topic-news": 8,
}
DST_TRAIN_PATH = "/home/ubuntu/data/sentence_classification/3_text/train.txt"
DST_TEST_PATH = "/home/ubuntu/data/sentence_classification/3_text/test.txt"
RATE = 0.7


def save(lines, start, end, dst_path):
    with open(dst_path, 'w') as fout:
        for (line, label) in lines[start: end]:
            fout.write("{}\t{}\n".format(line, label))


def load_file(label, path, lines):
    for line in open(path):
        line = line.strip()
        lines.append((line, label))


if __name__ == "__main__":
    lines = []
    for line in os.listdir(ROOT_DIR_PATH):
        path = os.path.join(ROOT_DIR_PATH, line)
        if line == "README.txt":
            continue
        if not os.path.isfile(path):
            continue
        head, tail = os.path.splitext(line)
        label = LABEL_MAP[head]
        load_file(label, path, lines)

    random.shuffle(lines)
    total_size = len(lines)
    train_size = int(RATE * total_size)
    test_size = total_size - train_size
    save(lines, 0, train_size, DST_TRAIN_PATH)
    save(lines, train_size, total_size, DST_TEST_PATH)
