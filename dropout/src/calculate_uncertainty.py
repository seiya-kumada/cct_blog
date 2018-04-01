#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from params import *  # noqa
from train import *  # noqa
import chainer
import numpy as np
import matplotlib.pyplot as plt
from data_maker import *  # noqa


def calculate_statistics(predictions):
    means = np.mean(predictions, axis=0)
    vars = np.var(predictions, axis=0) + 1.0 / TAU
    return (means, vars)


if __name__ == '__main__':
    # load the trained model
    model_path = os.path.join(OUTPUT_DIR_PATH, MODEL_NAME)
    model = MyNet(LENGTH_SCALE, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, DROPOUT_RATIO)
    chainer.serializers.load_npz(model_path, model)

    # predict values as if we were in training time
    predictions = np.empty((ITERATIONS, SAMPLING_SIZE), dtype=np.float32)
    xs = np.linspace(X_MIN, X_MAX, SAMPLING_SIZE)[:, np.newaxis].astype(np.float32)
    with chainer.using_config('train', True):  # you should be noticed that this is True!!
        for i in range(ITERATIONS):
            predictions[i] = model(xs).data.T

    plt.figure(figsize=(8, 5))

    # calculate means and variances
    means, vars = calculate_statistics(predictions)

    # draw uncertainty
    upper_bounds_3 = [mean + 3 * np.sqrt(var) for (mean, var) in zip(means, vars)]
    lower_bounds_3 = [mean - 3 * np.sqrt(var) for (mean, var) in zip(means, vars)]
    plt.fill_between(xs.reshape(-1, ), lower_bounds_3, upper_bounds_3, alpha=0.3, label='[-3σ,+3σ]', facecolor='red')

    # draw uncertainty
    upper_bounds_2 = [mean + 2 * np.sqrt(var) for (mean, var) in zip(means, vars)]
    lower_bounds_2 = [mean - 2 * np.sqrt(var) for (mean, var) in zip(means, vars)]
    plt.fill_between(xs.reshape(-1, ), lower_bounds_2, upper_bounds_2, alpha=0.3, label='[-2σ,+2σ]', facecolor='yellow')

    # draw uncertainty
    upper_bounds = [mean + np.sqrt(var) for (mean, var) in zip(means, vars)]
    lower_bounds = [mean - np.sqrt(var) for (mean, var) in zip(means, vars)]
    plt.fill_between(xs.reshape(-1, ), lower_bounds, upper_bounds, alpha=0.3, label='[-σ,+σ]', facecolor='green')

    # draw means
    plt.plot(xs, means, label='means')

    # draw original curve
    xs = np.linspace(X_MIN, X_MAX, SAMPLING_SIZE)
    ys = calculate_y(xs, MEAN, STDDEV, SHIFT)
    plt.plot(xs, ys, '--', label='original')

    # draw training dataset
    xs = np.load(XS_PATH)
    ys = np.load(YS_PATH)
    plt.scatter(xs, ys, label='training data')
    plt.ylim(0, 0.45)
    plt.legend(loc='lower center')
    plt.savefig('../results/result.png')
    plt.show()
