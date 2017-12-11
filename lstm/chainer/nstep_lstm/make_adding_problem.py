#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def mask(seq_size=200):
    mask = np.zeros(seq_size)
    indices = np.random.permutation(np.arange(seq_size))[:2]
    mask[indices] = 1
    return mask


def make_adding_problem(data_size, seq_size):
    masks = np.zeros((data_size, seq_size))
    for i in range(data_size):
        masks[i] = mask(seq_size)

    data = np.zeros((data_size, seq_size, 2))
    signals = np.random.uniform(low=0.0, high=1.0, size=(data_size, seq_size))
    data[:, :, 0] = signals[:]
    data[:, :, 1] = masks[:]
    target = (signals * masks).sum(axis=1).reshape(data_size, 1)
    return (data, target)


if __name__ == '__main__':
    data, target = make_adding_problem(data_size=10, seq_size=200)
    masks = data[:, :, 1]
    assert np.all(np.sum(masks, axis=1) == 2)
    assert data.shape == (10, 200, 2)
    assert target.shape == (10, 1)
