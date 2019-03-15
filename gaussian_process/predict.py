#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from pymc3.gp.util import plot_gp_dist
import pymc3 as pm
import numpy as np
import os
from params import *  # noqa


def plot_posterior(trc, xs, ys):
    # plot the results
    fig = plt.figure(figsize=(12, 5))
    ax = fig.gca()

    # plot the samples from the gp posterior with samples and shading
    # fの事後確率を描画
    plot_gp_dist(ax, trc["f"], xs)

    # plot the data and the true latent function
    plt.plot(xs, ys, 'ok', ms=3, alpha=0.5, label="Observed data")

    # axis labels and title
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc="best")
    plt.savefig(os.path.join(OUTPUT_DIR_PATH, "posterior.png"))


def calculate_posterior_predictive(model, gp, trace, start_x, end_x):
    # 200 new values from x=0 to x=15
    n_new = 200
    X_new = np.linspace(start_x, end_x, n_new)[:, None]

    # add the GP conditional to the model, given the new X values
    # 条件付き確率を作る。
    with model:
        f_pred = gp.conditional("f_pred", X_new)
        pred_samples = pm.sample_posterior_predictive(trace, vars=[f_pred], samples=1000)

    return pred_samples, X_new


def plot_posterior_predictive(pred_samples, X_new, X, y):
    fig = plt.figure(figsize=(24, 10))
    ax = fig.gca()
    plot_gp_dist(ax, pred_samples["f_pred"], X_new)
    plt.plot(X, y, 'ob', markersize=9, alpha=0.5, label="Observed data")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc="best")
    plt.savefig(os.path.join(OUTPUT_DIR_PATH, "pred_samples.png"))


def calculate_uncetainty(ps):
    f_pred = ps["f_pred"]
    row, col = f_pred.shape
    means = []
    stds = []
    for i in range(col):
        means.append(np.mean(f_pred[:, i]))
        stds.append(np.std(f_pred[:, i]))
    means = np.array(means)
    stds = np.array(stds)
    return means, stds


def plot_uncertainty(X, y, X_new, means, stds):
    plt.figure(figsize=(24, 10))
    plt.plot(X, y, 'ob', ms=9, alpha=0.5, label="Observed data")
    plt.plot(X_new, means, label="Prediction")
    plt.fill_between(X_new.squeeze(), means - stds, means + stds, color='red', alpha=0.3, label="$\sigma$")
    plt.xlabel("X")
    plt.ylabel("y)")
    plt.legend(loc="best")
    plt.savefig(os.path.join(OUTPUT_DIR_PATH, "uncertainty.png"))


def load_trace(dir_path, bX, by):
    with pm.Model() as model:  # noqa
        length = pm.Gamma("length", alpha=2, beta=1)
        eta = pm.HalfCauchy("eta", beta=5)

        cov = eta ** 2 * pm.gp.cov.Matern52(input_dim=1, ls=length)
        gp = pm.gp.Latent(cov_func=cov)

        f = gp.prior("f", X=bX)

        sigma = pm.HalfCauchy("sigma", beta=5)
        nu = pm.Gamma("nu", alpha=2, beta=0.1)
        y_ = pm.StudentT("y", mu=f, lam=1.0 / sigma, nu=nu, observed=by)  # noqa

        trace = pm.load_trace(dir_path)
        return model, gp, trace


if __name__ == "__main__":
    X = np.load(os.path.join(OUTPUT_DIR_PATH, "X.npy"))
    y = np.load(os.path.join(OUTPUT_DIR_PATH, "y.npy"))
    start_x = X[0, 0]
    start_y = X[-1, 0]

    model, gp, trace = load_trace(os.path.join(OUTPUT_DIR_PATH, "trace"), X, y)
    plot_posterior(trace, X, y)

    pred_samples, X_new = calculate_posterior_predictive(model, gp, trace, start_x, start_y)
    plot_posterior_predictive(pred_samples, X_new, X, y)

    means, stds = calculate_uncetainty(pred_samples)
    plot_uncertainty(X, y, X_new, means, stds)
