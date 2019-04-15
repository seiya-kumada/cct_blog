#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ARDRegression
np.random.seed(0)


def train(x, y):
    print("training ...")
    clf = ARDRegression(n_iter=1000, compute_score=True)
    clf.fit(x, y)
    print("training done!")
    return clf


def data_generator(x0, x1):
    return np.cos(0.5 * x0 * x1) + 0.1 * x0 * x1


def make_dataset(size):
    xs_0 = np.random.random(size)[:, np.newaxis]
    xs_1 = 100 * np.random.random(size)[:, np.newaxis]
    xs_2 = 0.1 * np.random.random(size)[:, np.newaxis]

    ys = np.array([data_generator(x0, x2) for (x0, x2) in zip(xs_0, xs_2)]).squeeze()
    xs = np.concatenate([xs_0, xs_1, xs_2], axis=1)
    return xs, ys


def display_weights(x, y, model):
    plt.figure(figsize=(6, 5))
    plt.title("Weights of the model")
    plt.plot(model.coef_, color="darkblue", alpha=0.5, marker="o", label="ARD estimate")
    plt.xlabel("Features")
    plt.ylabel("Values of the weights")
    plt.legend(loc="best")
    plt.savefig("./weights.jpg")

    for (i, w) in enumerate(model.coef_):
        print("[{:0>2}]: {}".format(i, w))


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
    xx = list(range(size))
    plt.figure(figsize=(13, 5))
    plt.title("Prediction")
    plt.errorbar(xx, py, std, fmt="ro", label="Prediction", marker="o")
    plt.scatter(xx, y, marker='o', color='blue', label="Ground Truth")
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

    mean = np.mean(error)
    std = np.std(error)
    print("mean: {}, std: {}".format(mean, std))


def normalize(x, y):
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    x = (x - x_mean) / x_std

    y_mean = np.mean(y, axis=0)
    y_std = np.std(y, axis=0)
    y = (y - y_mean) / y_std
    return x, y


def sample_0():
    observed_xs, observed_ys = make_dataset(40)
    observed_xs, observed_ys = normalize(observed_xs, observed_ys)

    model = train(observed_xs, observed_ys)
    display_weights(observed_xs, observed_ys, model)

    n_features = observed_xs.shape[1]
    display_weight_histogram(observed_xs, observed_ys, n_features, model)
    display_marginal_log_likelihood(model)

    py, std = predict(model, observed_xs, observed_ys)
    display_prediction(observed_ys, py, std)
    display_error(observed_ys, py)


if __name__ == "__main__":
    sample_0()
