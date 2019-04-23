#!/usr/bin/env python
# -*- coding: utf-8 -*-
import GPy
import numpy as np
import komatsu
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

K_FOLD_SIZE = 30


def display_predictions(i, pys, ys, mean, sigma, name):

    true_pys = pys * sigma + mean
    true_ys = ys * sigma + mean

    plt.figure(figsize=(15, 7))
    xs = range(true_pys.shape[0])
    plt.scatter(xs, true_ys, marker="o", label="Observed")
    plt.vlines(xs, ymin=0, ymax=510, linestyle="dashed", alpha=0.5)
    plt.xlabel("sample index")
    plt.ylabel("$y$")
    plt.scatter(xs, true_pys, label="Prediction", marker="o")
    plt.legend(loc="best")
    plt.savefig("./gp_komatsu/predictions_fold_{}_{}.jpg".format(name, i))
    plt.close()


def display_weights(i, model):
    ls = list(model.kern.lengthscale)
    weights = [1 / v for v in ls]

    plt.figure(figsize=(15, 6))
    plt.ylabel("$1/l_m$")
    plt.xlabel("$m$")
    xs = list(range(len(weights)))
    plt.scatter(xs, weights, marker="o", label="weight")
    plt.xticks(list(range(n_features)))
    plt.vlines(list(range(n_features)), ymax=5, ymin=0, alpha=0.2, linestyle="dashed")

    plt.legend(loc="best")
    plt.savefig("./gp_komatsu/weights_fold_{}.jpg".format(i))
    plt.close()


# def evaluate(ys, pys, name):
#     errors = np.abs(pys - ys)
#     error_mean = np.mean(errors)
#     error_std = np.std(errors)
#     print("{} mean: {} std: {}".format(name, error_mean, error_std))


def evaluate_(ys, pys, y_mean, y_std, name):
    true_ys = ys * y_std + y_mean
    true_pys = pys * y_std + y_mean

    errors = np.abs(true_pys - true_ys) / true_ys
    error_mean = np.mean(errors)
    # error_std = np.std(errors)
    # print("{} mean: {} std: {}".format(name, error_mean, error_std))
    return error_mean


if __name__ == "__main__":
    hasaki_names, hasaki = komatsu.load_data(komatsu.HASAKI_NAMES, komatsu.HASAKI)
    mamouryo_names, mamouryo = komatsu.load_data(komatsu.MAMOURYO_NAMES, komatsu.MAMOURYO)

    ys = mamouryo[1]
    xs = hasaki.transpose(1, 0)

    raw_ys = ys.copy()
    raw_xs = xs.copy()

    # normalize dataset
    xs, ys, xstats, ystats = komatsu.normalize(xs, ys)
    x_mean, x_std = xstats
    y_mean, y_std = ystats

    ys = ys[:, np.newaxis]

    kf = KFold(n_splits=K_FOLD_SIZE, shuffle=False, random_state=1)

    # define kernels
    n_features = xs.shape[1]
    kernel = GPy.kern.Matern52(n_features, ARD=True)

    test_errors = []
    for i, (train_indices, test_indices) in enumerate(kf.split(xs)):
        train_xs = xs[train_indices]
        train_ys = ys[train_indices]
        model = GPy.models.GPRegression(train_xs, train_ys, kernel)
        model.optimize(messages=False, max_iters=1e5)

        test_xs = xs[test_indices]
        test_ys = ys[test_indices]

        display_weights(i, model)

        prediction_train_ys, _ = model.predict(train_xs)
        # evaluate(train_ys, prediction_train_ys, "train")
        evaluate_(train_ys, prediction_train_ys, y_mean, y_std, "train")

        prediction_test_ys, prediction_test_std = model.predict(test_xs)
        # evaluate(test_ys, prediction_test_ys,  "test")
        error = evaluate_(test_ys, prediction_test_ys, y_mean, y_std, "test")

        display_predictions(i, prediction_train_ys, train_ys, y_mean, y_std, "train")
        display_predictions(i, prediction_test_ys, test_ys, y_mean, y_std, "test")
        test_errors.append(error)

    print(np.mean(test_errors))
