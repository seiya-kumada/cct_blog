#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import ARDRegression
np.random.seed(0)


DIR_PATH = "/Users/kumada/Data/小松/2019_03_15"
HASAKI_NAMES = "hasaki_names.npy"
HASAKI = "hasaki.npy"
MAMOURYO_NAMES = "mamouryo_names.npy"
MAMOURYO = "mamouryo.npy"
SESSAKU_NAMES = "sessaku_names.npy"
SESSAKU = "sessaku.npy"


def train(x, y):
    print("training ...")
    clf = ARDRegression(compute_score=True)
    clf.fit(x, y)
    print("training done!")
    return clf


def load_data(names_path, data_path):
    names = np.load(os.path.join(DIR_PATH, names_path))
    data = np.load(os.path.join(DIR_PATH, data_path))
    return names, data


def display_weights(x, y, model):
    plt.figure(figsize=(6, 5))
    plt.title("Weights of the model")
    plt.plot(model.coef_, color="darkblue", alpha=0.5, marker="o", label="ARD estimate")
    plt.xlabel("Features")
    plt.ylabel("Values of the weights")
    plt.legend(loc="best")
    plt.savefig("./weights.jpg")

    for (i, w) in enumerate(model.coef_):
        print("{:0>2}: {}".format(i, w))


def display_weight_histogram(x, y, n_features, model):
    plt.figure(figsize=(6, 5))
    plt.title("Histogram of the weights")
    plt.hist(model.coef_, bins=n_features, color='navy', log=True)
    plt.ylabel("Features")
    plt.xlabel("Values of the weights")
    plt.legend(loc="best")
    plt.savefig("./histogram.jpg")


def display_marginal_log_likelihood(model):
    plt.figure(figsize=(6, 5))
    plt.title("Marginal log-likelihood")
    plt.plot(model.scores_, color='navy', linewidth=2)
    plt.ylabel("Score")
    plt.xlabel("Iterations")
    plt.legend(loc="best")
    plt.savefig("./mll.jpg")


def predict(model, x, y):
    size, = y.shape
    py, std = model.predict(x, return_std=True)
    return py, std


def display_prediction(y, py, std):
    size, = y.shape
    plt.figure(figsize=(6, 5))
    plt.title("Prediction")
    plt.plot(y, marker='o', color='blue', label="Ground Truth")
    plt.plot(py, marker='o', color='red', label="Prediction")
    plt.ylabel("$y$")
    plt.xlabel("$x$")
    plt.legend(loc="best")
    plt.savefig("./prediction.jpg")


def display_error(y, py):
    plt.figure(figsize=(6, 5))
    plt.title("Prediction")
    error = np.abs(y - py) / y
    plt.plot(error, marker='o', color='blue')
    plt.ylabel("error")
    plt.xlabel("$x$")
    plt.legend(loc="best")
    plt.savefig("./error.jpg")


def sample_0():
    hasaki_names, hasaki = load_data(HASAKI_NAMES, HASAKI)
    mamouryo_names, mamouryo = load_data(MAMOURYO_NAMES, MAMOURYO)

    y = mamouryo[0]
    X = np.transpose(hasaki, (1, 0))
    print("(X.shape,y.shape)=({},{})".format(X.shape, y.shape))
    model = train(X, y)
    display_weights(X, y, model)

    n_features = X.shape[1]
    display_weight_histogram(X, y, n_features, model)
    display_marginal_log_likelihood(model)

    py, std = predict(model, X, y)
    display_prediction(y, py, std)
    display_error(y, py)
    print(std)


if __name__ == "__main__":
    sample_0()
