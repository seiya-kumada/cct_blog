#!/usr/bin/env python
# -*- coding: utf-8 -*-
from params import *  # noqa
import numpy as np
import matplotlib.pyplot as plt
import utils


def ground_truth(x):
    return x + np.sin(3 * x)


def draw_uncertainty(oxs, oys, sigmas, n, color):
    upper_bounds = oys + n * sigmas
    lower_bounds = oys - n * sigmas
    plt.fill_between(oxs, lower_bounds, upper_bounds, alpha=0.3, label='[-{n}σ,+{n}σ]'.format(n=n), facecolor=color)


if __name__ == '__main__':
    ymeans = np.load(YMEANS_PATH)
    ystds = np.load(YSTDS_PATH)
    ixs = np.load(IXS_PATH)
    ypredictions = np.load(YPREDICTIONS_PATH)

    plt.title('Bayesian Inference')

    # draw a predictive curve
    plt.plot(ixs, ymeans, label='predictive curve', linestyle='dashed')

    # draw an original curve
    plt.plot(ixs, ground_truth(ixs), label='original curve')

    # draw observed dataset
    dataset = utils.load_dataset(DATASET_PATH)
    observed_xs = dataset[:, 0]
    observed_ys = dataset[:, 1]
    plt.scatter(observed_xs, observed_ys, label='observed dataset')

    # draw uncertainties
    draw_uncertainty(ixs, ymeans, ystds, 3, 'red')
    draw_uncertainty(ixs, ymeans, ystds, 2, 'yellow')
    draw_uncertainty(ixs, ymeans, ystds, 1, 'green')

    plt.ylim(-0.1, 5.5)
    plt.legend(loc='best')
    plt.savefig('results/mcbayes_{}.png'.format(M))
