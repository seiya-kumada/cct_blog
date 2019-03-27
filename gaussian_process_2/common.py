#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pymc3 as pm
import theano.tensor as tt
import theano


def data_generator(x):
    return np.cos(0.5 * x * x) + 0.1 * x * x


def generate_observed_dataset():
    a = np.linspace(0, 1, 10)
    b = np.linspace(3, 6, 20)
    c = np.linspace(6, 10, 10)
    observed_xs = np.sort(np.concatenate([a, b, c]))
    observed_ys = [data_generator(x) + np.random.normal(loc=0, scale=0.2) for x in observed_xs]
    real_xs = np.linspace(0, 10, 200)
    real_ys = [data_generator(x) for x in real_xs]
    return observed_xs[:, None], np.array(observed_ys), real_xs, np.array(real_ys)


def generate_real_dataset():
    real_xs = np.linspace(0, 10, 100)
    real_ys = [data_generator(x) for x in real_xs]
    return (real_xs, real_ys)


def generate_observed_dataset_2():
    np.random.seed(20090425)
    n = 20
    X = np.sort(3 * np.random.rand(n))[:, None]

    with pm.Model() as model:  # noqa
        # f(x)
        l_true = 0.3
        s2_f_true = 1.0
        cov = s2_f_true * pm.gp.cov.ExpQuad(1, l_true)

        # noise, epsilon
        s2_n_true = 0.1
        K_noise = s2_n_true ** 2 * tt.eye(n)
        K = cov(X) + K_noise

        # evaluate the covariance with the given hyperparameters
        K = theano.function([], cov(X) + K_noise)()

        # generate fake data from GP with white noise (with variance sigma2)
        y = np.random.multivariate_normal(np.zeros(n), K)
    return X, y
