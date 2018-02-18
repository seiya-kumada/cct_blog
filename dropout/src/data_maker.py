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


# draw graph
def draw_graph(x_min, x_max, mean, stddev, shift):
    xs = np.linspace(x_min, x_max, 300)
    ys = calculate_y(xs, mean, stddev, shift)

    plt.plot(xs, ys)
    plt.show()


# generate dataset
def generate_dataset(sample_size, input_dim, x_min, x_max, mean, stddev, shift):
    xs = np.zeros((sample_size, input_dim), dtype=np.float32)
    ys = np.zeros((sample_size, input_dim), dtype=np.float32)
    for i in range(sample_size):
        xs[i] = np.random.uniform(low=x_min, high=x_max, size=(input_dim,))
        ys[i] = calculate_y(xs[i], mean, stddev, shift)
    return xs, ys


# draw dataset
def draw_dataset(xs, ys, x_min, x_max, mean, stddev, shift):
    sample_size, _ = xs.shape
    i = np.random.randint(low=0, high=sample_size)
    x = xs[i]
    y = ys[i]
    plt.scatter(x, y)
    xs = np.linspace(x_min, x_max, 300)
    ys = calculate_y(xs, mean, stddev, shift)
    plt.plot(xs, ys)
    plt.show()


if __name__ == '__main__':
    # draw_graph(X_MIN, X_MAX, MEAN, STDDEV, SHIFT)

    xs, ys = generate_dataset(SAMPLE_SIZE, INPUT_DIM, X_MIN, X_MAX, MEAN, STDDEV, SHIFT)
    np.save(XS_PATH, xs)
    np.save(YS_PATH, ys)

    # draw dataset
    xs = np.load(XS_PATH)
    ys = np.load(YS_PATH)
    print(xs.dtype)
    print(ys.dtype)
    # draw_dataset(xs, ys, X_MIN, X_MAX, MEAN, STDDEV, SHIFT)
