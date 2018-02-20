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
    unknown_size = 100
    # unknown_xs = np.random.uniform(X_MIN, X_MAX, size=unknown_size).astype(np.float32).reshape(-1, 1)
    unknown_xs = np.linspace(X_MIN, X_MAX, unknown_size)[:, np.newaxis].astype(np.float32)
    print(unknown_xs.shape)
    # predict ys
    with chainer.using_config('train', False):
        predictive_ys = model(unknown_xs)

    plt.scatter(unknown_xs, predictive_ys.data, label='predictive for test data')


def sample_1(model):
    xs = np.load(XS_PATH)
    ys = np.load(YS_PATH)

    with chainer.using_config('train', False):
        pred_ys = model(xs)

    plt.figure(figsize=(10, 5))
    plt.scatter(xs, ys, label='training data')
    plt.scatter(xs, pred_ys.data, label='predictive for training data')
    xs = np.linspace(X_MIN, X_MAX, 300)
    ys = calculate_y(xs, MEAN, STDDEV, SHIFT)
    plt.plot(xs, ys, label='original')

    sample_0(model)
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':

    # load the trained model
    model_path = os.path.join(OUTPUT_DIR_PATH, MODEL_NAME)
    model = MyNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, DROPOUT_RATIO)
    chainer.serializers.load_npz(model_path, model)

    sample_1(model)
