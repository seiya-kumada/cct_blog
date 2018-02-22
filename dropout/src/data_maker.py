#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
from params import *  # noqa

np.random.seed(1)


# two gaussians
def calculate_y(x, mean, stddev, shift):
    return norm.pdf(x=x, loc=mean + shift, scale=stddev) + norm.pdf(x=x, loc=mean - shift, scale=stddev)


if __name__ == '__main__':
    # generate dataset
    xs = np.random.uniform(X_MIN + 4, X_MAX - 4, size=SAMPLE_SIZE).astype(np.float32)  # .reshape(-1, 1)
    # xs = np.linspace(X_MIN, X_MAX, SAMPLE_SIZE)
    ys = calculate_y(xs, MEAN, STDDEV, SHIFT)
    # noise = np.random.normal(0, NOISE_STDDEV, SAMPLE_SIZE)
    # ys = ys + noise
    xs = xs.reshape(-1, 1).astype(np.float32)
    ys = ys.reshape(-1, 1).astype(np.float32)

    # save them
    np.save(XS_PATH, xs)
    np.save(YS_PATH, ys)

    # # draw dataset
    xs = np.load(XS_PATH)
    ys = np.load(YS_PATH)
    plt.scatter(xs, ys)
    xs = np.linspace(X_MIN, X_MAX, 300)
    ys = calculate_y(xs, MEAN, STDDEV, SHIFT)
    plt.plot(xs, ys)
    plt.show()
