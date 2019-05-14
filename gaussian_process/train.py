#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import pymc3 as pm
from params import *  # noqa


def read_dataset():
    hasaki_names = np.load(os.path.join(DIR_PATH, HASAKI_NAMES))
    hasaki = np.load(os.path.join(DIR_PATH, HASAKI))
    mamouryo_names = np.load(os.path.join(DIR_PATH, MAMOURYO_NAMES))
    mamouryo = np.load(os.path.join(DIR_PATH, MAMOURYO))
    sessaku_names = np.load(os.path.join(DIR_PATH, SESSAKU_NAMES))
    sessaku = np.load(os.path.join(DIR_PATH, SESSAKU))
    return hasaki_names, hasaki, mamouryo_names, mamouryo, sessaku_names, sessaku


def sort_dataset_along_X(rX, ry):
    X_index = np.argsort(rX, axis=0)
    X = rX[X_index].squeeze()[:, None]
    y = ry[X_index].squeeze()
    return (X, y)


def normalize_y(ay):
    y_max = np.max(ay)
    y_min = np.min(ay)

    def scale(v):
        return 2 * (v - y_min) / (y_max - y_min) - 1

    y = [scale(v) for v in ay]
    y_mean = np.mean(y)
    y_std = np.std(y)
    y = [(v - y_mean) / y_std for v in y]
    return y


def execute_mcmc(bX, by):
    with pm.Model() as model:  # noqa
        length = pm.Gamma("length", alpha=2, beta=1)
        eta = pm.HalfCauchy("eta", beta=5)

        cov = eta ** 2 * pm.gp.cov.Matern52(input_dim=1, ls=length)
        gp = pm.gp.Latent(cov_func=cov)

        f = gp.prior("f", X=bX)

        sigma = pm.HalfCauchy("sigma", beta=5)
        nu = pm.Gamma("nu", alpha=2, beta=0.1)
        y_ = pm.StudentT("y", mu=f, lam=1.0 / sigma, nu=nu, observed=by)  # noqa

        trace = pm.sample(1000)
        return trace


if __name__ == "__main__":
    hasaki_names, hasaki, mamouryo_names, mamouryo, sessaku_names, sessaku = read_dataset()
    raw_y = mamouryo[MAMOURYO_INDEX]
    raw_X = sessaku[SESSAKU_INDEX][:, None]

    X, y = sort_dataset_along_X(raw_X, raw_y)
    y = normalize_y(y)

    if not os.path.isdir(OUTPUT_DIR_PATH):
        os.makedirs(OUTPUT_DIR_PATH)

    np.save(os.path.join(OUTPUT_DIR_PATH, "X.npy"), X)
    np.save(os.path.join(OUTPUT_DIR_PATH, "y.npy"), y)

    trace = execute_mcmc(X, y)
    pm.save_trace(trace, os.path.join(OUTPUT_DIR_PATH, "trace"), overwrite=True)
    print(pm.gelman_rubin(trace))
