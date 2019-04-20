#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import sklearn_ardregression
from sklearn.linear_model import ARDRegression
np.random.seed(0)
plt.rcParams["font.size"] = 18


def train(x, y):
    print("training ...")
    clf = ARDRegression(n_iter=1000, compute_score=True)
    clf.fit(x, y)
    print("training done!")
    return clf


def display_weights(model):
    plt.figure(figsize=(7, 6))
    plt.ylabel("$w_m$")
    plt.xlabel("$m$")
    xs = list(range(len(model.coef_)))
    plt.errorbar(xs, model.coef_, np.diag(model.sigma_), fmt="ro", marker="o", label="weight")
    plt.xticks([0, 1, 2])
    plt.legend(loc="lower left")
    plt.savefig("./images/weights.jpg")


def display_weight_histogram(x, y, n_features, model):
    plt.figure(figsize=(6, 5))
    plt.title("Histogram of the weights")
    plt.hist(model.coef_, bins=n_features, color='navy', log=True)
    plt.ylabel("Features")
    plt.xlabel("Values of the weights")
    plt.legend(loc="best")
    plt.savefig("./images/histogram.jpg")


def display_marginal_log_likelihood(model):
    plt.figure(figsize=(6, 5))
    plt.title("Marginal log-likelihood")
    plt.plot(model.scores_, color='navy', linewidth=2)
    plt.ylabel("Score")
    plt.xlabel("Iterations")
    plt.legend(loc="best")
    plt.savefig("./images/mll.jpg")


def predict(model, x):
    py, std = model.predict(x, return_std=True)
    return py, std


def display_prediction(y, py, std, name, hlines=False):
    plt.figure(figsize=(15, 7))
    xs = range(y.shape[0])

    plt.scatter(xs, y, marker="o", label="Observed")
    if hlines:
        plt.vlines(list(range(y.shape[0])), ymin=-3, ymax=3, linestyle="dashed", alpha=0.5)
    plt.xlabel("sample index")
    plt.ylabel("$y$")
    plt.ylim(-3, 3)
    plt.errorbar(xs, py, std, fmt="ro", label="Prediction", marker="o")
    plt.legend(loc="best")
    plt.savefig(name)


def display_error(y, py, name):
    abs_errors = np.abs(y - py)

    plt.figure(figsize=(13, 5))
    size = abs_errors.shape[0]
    xs = list(range(size))
    plt.scatter(xs, abs_errors, marker=".")
    plt.xlabel("sample index")
    plt.ylabel("absolute value of diff")
    plt.savefig(name)

    mean_abs = np.mean(abs_errors)
    std_abs = np.std(abs_errors)
    print("差分の絶対値の平均値: {}".format(mean_abs))
    print("差分の絶対値の標準偏差: {}".format(std_abs))


def normalize(x, y):
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    x = (x - x_mean) / x_std

    y_mean = np.mean(y, axis=0)
    y_std = np.std(y, axis=0)
    y = (y - y_mean) / y_std
    return x, y


def sample_0():
    # 訓練データ
    TRAIN_DATA_SIZE = 100
    train_xs, train_ys = sklearn_ardregression.make_dataset(TRAIN_DATA_SIZE)
    train_xs, train_ys = sklearn_ardregression.normalize(train_xs, train_ys)

    model = train(train_xs, train_ys)

    # 重みを見る。
    display_weights(model)
    display_marginal_log_likelihood(model)

    # 訓練時の精度を見る。
    py, std = predict(model, train_xs)
    display_prediction(train_ys, py, std, "./images/train_accuracy.jpg")
    display_error(train_ys, py, "./images/train_error.jpg")

    # テストデータ
    TEST_DATA_SIZE = 30
    test_xs, test_ys = sklearn_ardregression.make_dataset(TEST_DATA_SIZE)
    test_xs, test_ys = sklearn_ardregression.normalize(test_xs, test_ys)

    # テスト時の精度を見る。
    py, std = predict(model, test_xs)
    display_prediction(test_ys, py, std, "./images/test_accuracy.jpg", hlines=True)
    display_error(test_ys, py, "./images/test_error.jpg")


if __name__ == "__main__":
    sample_0()
