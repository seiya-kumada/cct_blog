#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pickle
import params
import numpy as np


def make_line(label, row):
    line = str(label)
    for i, v in enumerate(row, start=1):
        if abs(v) < 1.0e-6:
            continue
        line += " {}:{}".format(i, v)
    return line


if __name__ == "__main__":
    labels = pickle.load(open(params.LABEL_PATH, 'rb'))
    print(len(labels))

    mat = np.load(params.EMBEDDINGS_PATH)
    print(mat.shape)

    with open(params.LIBSVM_PATH, 'w') as fout:
        for label, row in zip(labels, mat):
            line = make_line(label, row)
            fout.write("{}\n".format(line))
