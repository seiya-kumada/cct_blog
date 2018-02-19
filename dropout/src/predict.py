#!/usr/bin/env python
# -*- coding: utf-8 -*-

from train import *  # noqa
import chainer
from params import *  # noqa
import os
from data_maker import *  # noqa
import matplotlib.pyplot as plt


if __name__ == '__main__':
    model_path = os.path.join(OUTPUT_DIR_PATH, MODEL_NAME)
    model = MyNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, DROPOUT_RATIO)
    chainer.serializers.load_npz(model_path, model)

    sample_size = 10
    sample_xs, sample_ys = generate_dataset(sample_size, INPUT_DIM, X_MIN, X_MAX, MEAN, STDDEV, SHIFT)

    with chainer.using_config('train', False):
        pred_ys = model(sample_xs)

    for i in range(sample_size):
        # plt.scatter(sample_xs[i], sample_ys[i])
        plt.scatter(sample_xs[i], pred_ys.data[i])

    xs = np.linspace(X_MIN, X_MAX, 300)
    ys = calculate_y(xs, MEAN, STDDEV, SHIFT)
    plt.plot(xs, ys)
    plt.ylim(-0.1, 0.5)
    plt.show()
