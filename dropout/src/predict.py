#!/usr/bin/env python
# -*- coding: utf-8 -*-

from train import *  # noqa
import chainer
from params import *  # noqa
import os
from data_maker import *  # noqa
import matplotlib.pyplot as plt


def sample_0(model):
    # generate random values within [X_MIN, X_MAX]
    unknown_size = 10
    unknown_xs = np.random.uniform(X_MIN, X_MAX, size=unknown_size).astype(np.float32).reshape(-1, 1)

    # predict ys
    with chainer.using_config('train', False):
        predictive_ys = model(unknown_xs)

    plt.scatter(unknown_xs, predictive_ys.data)
    xs = np.linspace(X_MIN, X_MAX, 300)
    ys = calculate_y(xs, MEAN, STDDEV, SHIFT)
    plt.plot(xs, ys)
    plt.show()


def sample_1(model):
    xs = np.load(XS_PATH)
    ys = np.load(YS_PATH)

    with chainer.using_config('train', False):
        pred_ys = model(xs)

    plt.scatter(xs, ys)
    plt.scatter(xs, pred_ys.data)
    xs = np.linspace(X_MIN, X_MAX, 300)
    ys = calculate_y(xs, MEAN, STDDEV, SHIFT)
    plt.plot(xs, ys)
    plt.show()


if __name__ == '__main__':

    # load the trained model
    model_path = os.path.join(OUTPUT_DIR_PATH, MODEL_NAME)
    model = MyNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, DROPOUT_RATIO)
    chainer.serializers.load_npz(model_path, model)

    sample_1(model)
