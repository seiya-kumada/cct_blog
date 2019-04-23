#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import ARDRegression
np.random.seed(0)


DIR_PATH = "/Users/kumada/Data/小松/追加解析/task2_1"
# DIR_PATH = "/Users/kumada/Data/小松/2019_03_15"
HASAKI_NAMES = "hasaki_names.npy"
HASAKI = "hasaki.npy"
MAMOURYO_NAMES = "mamouryo_names.npy"
MAMOURYO = "mamouryo.npy"
SESSAKU_NAMES = "sessaku_names.npy"
SESSAKU = "sessaku.npy"


def train(x, y):
    print("training ...")
    clf = ARDRegression(n_iter=1000, compute_score=True)
    clf.fit(x, y)
    print("training done!")
    return clf


def load_data(names_path, data_path):
    names = np.load(os.path.join(DIR_PATH, names_path))
    data = np.load(os.path.join(DIR_PATH, data_path))
    return names, data


def display_weights(x, y, hasaki_names, model):
    plt.figure(figsize=(6, 5))
    plt.title("Weights of the model")
    plt.plot(model.coef_, color="darkblue", alpha=0.5, marker="o", label="ARD estimate")
    plt.xlabel("Features")
    plt.ylabel("Values of the weights")
    plt.legend(loc="best")
    plt.savefig("./komatsu/weights.jpg")

    # for (i, w) in enumerate(model.coef_):
    #     print("[{:0>2}]{}: {}".format(i, hasaki_names[i], w))


def display_weight_histogram(x, y, n_features, model):
    plt.figure(figsize=(6, 5))
    plt.title("Histogram of the weights")
    plt.hist(model.coef_, bins=n_features, color='navy', log=True)
    plt.ylabel("Features")
    plt.xlabel("Values of the weights")
    plt.legend(loc="best")
    plt.savefig("./komatsu/histogram.jpg")


def display_marginal_log_likelihood(model):
    plt.figure(figsize=(6, 5))
    plt.title("Marginal log-likelihood")
    plt.plot(model.scores_, color='navy', linewidth=2)
    plt.ylabel("Score")
    plt.xlabel("Iterations")
    plt.legend(loc="best")
    plt.savefig("./komatsu/mll.jpg")


def predict(model, x, y):
    size, = y.shape
    py, std = model.predict(x, return_std=True)
    return py, std


def display_prediction(y, py, std):
    size, = y.shape
    plt.figure(figsize=(6, 5))
    plt.title("Prediction")
    xs = list(range(len(y)))
    plt.scatter(xs, y, marker='.', color='blue', label="Ground Truth")
    plt.scatter(xs, py, marker='.', color='red', label="Prediction")
    plt.vlines(xs, ymin=-2, ymax=2, linestyle="dashed", alpha=0.2, linewidth=1)
    plt.ylabel("$y$")
    plt.xlabel("$x$")
    plt.legend(loc="best")
    plt.savefig("./komatsu/prediction.jpg")
    errors = np.abs(y - py)
    mean = np.mean(errors)
    std = np.std(errors)
    print("差分の絶対値の平均値: {}".format(mean))
    print("差分の絶対値の標準偏差: {}".format(std))


def display_error(y, py):
    plt.figure(figsize=(6, 5))
    plt.title("Prediction")
    error = np.abs(y - py) / y
    plt.plot(error, marker='o', color='blue')
    plt.ylabel("error")
    plt.xlabel("$x$")
    plt.legend(loc="best")
    plt.savefig("./komatsu/error.jpg")


def normalize(x, y):
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    x = (x - x_mean) / x_std

    y_mean = np.mean(y, axis=0)
    y_std = np.std(y, axis=0)
    y = (y - y_mean) / y_std
    return x, y, (x_mean, x_std), (y_mean, y_std)


def sample_0():
    hasaki_names, hasaki = load_data(HASAKI_NAMES, HASAKI)
    mamouryo_names, mamouryo = load_data(MAMOURYO_NAMES, MAMOURYO)

    y = mamouryo[1]
    X = np.transpose(hasaki, (1, 0))

    print("AAA", y.shape, X.shape)
    X, y = normalize(X, y)
    print(X.shape, y.shape)
    model = train(X, y)
    display_weights(X, y, hasaki_names, model)

    n_features = X.shape[1]
    display_weight_histogram(X, y, n_features, model)
    display_marginal_log_likelihood(model)

    py, std = predict(model, X, y)
    display_prediction(y, py, std)
    display_error(y, py)


if __name__ == "__main__":
    sample_0()
