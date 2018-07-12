#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pymc

DATASET_PATH = './dataset.txt'
M = 3
ALPHA = 0.1
SIGMA = 0.015
TAU = 1 / SIGMA**2


def load_dataset(path):
    dataset = []
    for line in open(path):
        x, y = line.strip().split()
        dataset.append([float(x), float(y)])
    return np.array(dataset)


def calculate_ground_truth(x):
    return x + np.sin(3 * x)


def plot_ground_truth():
    xs = np.linspace(-1, 4.5)[:, None]
    ys = calculate_ground_truth(xs)
    plt.figure(figsize=(8, 8))
    plt.plot(xs, ys)
    plt.axis('equal')
    plt.scatter(observed_xs, observed_ys)
    plt.ylim(-0.5, 4.5)
    plt.xlim(-0.5, 4.5)
    plt.grid(True)
    plt.show()


# @pymc.deterministic
# def linear_regression(xs=xs, ws=ws):
#     return ws.dot(xs)


if __name__ == '__main__':

    # load a dataset
    dataset = load_dataset(DATASET_PATH)
    observed_xs = dataset[:, 0]
    observed_ys = dataset[:, 1]

    # plot_ground_truth()

    # define a prior for w, which is a multivariable gaussian
    ws = np.empty((M, 1), dtype=object)
    for i in range(M):
        ws[i] = pymc.Normal('w{}'.format(i), mu=0, tau=ALPHA, value=0)
    ws = ws.T

    # calculate x^0, x^1, ..., x^{M-1}
    xs = np.empty((M, len(observed_xs)), dtype=np.float32)
    for i in range(M):
        xs[i] = np.power(observed_xs, i)

    assert(ws.shape == (1, M))
    assert(xs.shape == (M, len(observed_xs)))

    # define a deterministic function
    linear_regression = pymc.Lambda('linear_regression', lambda xs=xs, ws=ws: ws.dot(xs))

    # define a model likelihood
    # y = pymc.Normal('y', mu=linear_regression, tau=TAU, value=observed_ys, observed=True)
