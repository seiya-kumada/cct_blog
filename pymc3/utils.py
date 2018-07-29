#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


def load_dataset(path):
    dataset = []
    for line in open(path):
        x, y = line.strip().split()
        dataset.append([float(x), float(y)])
    return np.array(dataset)
