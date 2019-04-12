#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import ARDRegression
np.random.seed(0)


def generate_dataset(n_samples, n_features):

    # standard gaussian distribution
    X = np.random.randn(n_samples, n_features)

    lambda_ = 4.0
    w = np.zeros(n_features)

    # generate 10 integers within [0,n_features)
    relevant_features = np.random.randint(0, n_features, 10)

    for i in relevant_features:
        # loc and scale mean an average and a standard deviation, respectively.
        w[i] = stats.norm.rvs(loc=0, scale=1.0 / np.sqrt(lambda_))

    alpha_ = 50
    noise = stats.norm.rvs(loc=0, scale=1.0 / np.sqrt(alpha_), size=n_samples)

    y = np.dot(X, w) + noise
    return X, y, w, relevant_features


def train(x, y):
    clf = ARDRegression(compute_score=True)
    clf.fit(x, y)
    return clf


def display_weights(x, y, w, model):
    plt.figure(figsize=(6, 5))
    plt.title("Weights of the model")
    plt.plot(model.coef_, color="darkblue", alpha=0.5, linestyle="-", linewidth=2, label="ARD estimate")
    if w is not None:
        plt.plot(w, color='orange', alpha=0.5, linestyle='-', linewidth=2, label="Ground truth")
    plt.xlabel("Features")
    plt.ylabel("Values of the weights")
    plt.legend(loc=1)
    plt.show()


def display_weight_histogram(x, y, n_features, relevant_features, model):
    plt.figure(figsize=(6, 5))
    plt.title("Histogram of the weights")
    plt.hist(model.coef_, bins=n_features, color='navy', log=True)
    plt.scatter(model.coef_[relevant_features], np.full(len(relevant_features), 5.),
                color='gold', marker='o', label="Relevant features")
    plt.ylabel("Features")
    plt.xlabel("Values of the weights")
    plt.legend(loc=1)
    plt.show()


def display_marginal_log_likelihood(model):
    plt.figure(figsize=(6, 5))
    plt.title("Marginal log-likelihood")
    plt.plot(model.scores_, color='navy', linewidth=2)
    plt.ylabel("Score")
    plt.xlabel("Iterations")
    plt.legend(loc=1)
    plt.show()


def f(x, noise_amount):
    y = np.sqrt(x) * np.sin(x)
    noise = np.random.normal(0, 1, len(x))
    return y + noise_amount * noise


def display(y_mean, y_std, X_plot, y_plot):
    plt.figure(figsize=(6, 5))
    plt.errorbar(X_plot, y_mean, y_std, color="navy", label="Polynomial ARD", linewidth=2)
    plt.plot(X_plot, y_plot, color="gold", linewidth=2, label="Ground Truth")
    plt.ylabel("Output y")
    plt.xlabel("Feature X")
    plt.legend(loc="lower left")
    plt.show()


def train_(X, y, degree):
    model = ARDRegression(threshold_lambda=1e5)
    X = np.vander(X, degree)
    model.fit(X, y)
    return model


if __name__ == "__main__":
    # 100 samples, each of which has 100-dimension
    n_samples, n_features = 50, 100
    X, y, w, relevant_features = generate_dataset(n_samples, n_features)
    print(X.shape, y.shape)
    model = train(X, y)

    # display_weights(X, y, w, model)
    # display_weight_histogram(X, y, n_features, relevant_features, clf)
    # display_marginal_log_likelihood(clf)

    # n_samples = 100
    # degree = 10
    # X = np.linspace(0, 10, n_samples)
    # y = f(X, noise_amount=1)
    # model = train_(X, y, degree)
    # print(model.coef_)
    # display_weight_histogram(X, y, degree, [8], model)

    # display_weights(X, y, None, model)

    # X_plot = np.linspace(0, 11, 25)
    # y_plot = f(X_plot, noise_amount=0)
    # y_mean, y_std = model.predict(np.vander(X_plot, degree), return_std=True)
    # display(y_mean, y_std, X_plot, y_plot)
