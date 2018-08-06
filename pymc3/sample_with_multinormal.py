#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import utils
import pymc3 as pymc
from params import *  # noqa
import matplotlib.pyplot as plt
import pickle


if __name__ == '__main__':

    # load a dataset
    dataset = utils.load_dataset(DATASET_PATH)
    observed_xs = dataset[:, 0]
    observed_ys = dataset[:, 1]
    data_size, = observed_xs.shape

    # calculate x^0, x^1, ..., x^{M-1}
    xs = np.empty((M, len(observed_xs)), dtype=np.float32)
    for i in range(M):
        xs[i] = np.power(observed_xs, i)
    assert(xs.shape == (M, len(observed_xs)))

    with pymc.Model() as model:
        # define a prior for w, which is a multivariable gaussian
        # ws = pymc.MvNormal('ws', mu=np.zeros(M), tau=ALPHA * np.eye(M), testval=np.zeros(M), shape=(data_size, M))
        ws = pymc.MvNormal('ws', mu=np.zeros(M), tau=ALPHA * np.eye(M), shape=M, testval=np.zeros(M))

        # define likelihood
        linear_regression = pymc.Deterministic('linear_regression', ws.dot(xs))
        y = pymc.Normal('y', mu=linear_regression, tau=TAU, observed=observed_ys)

        start = pymc.find_MAP()
        step = pymc.NUTS(max_treedepth=30)  # , target_accept=0.95)
        trace = pymc.sample(500, tune=1000, step=step, start=start)  # , step, start, tune=1000)

    with open('my_model.pkl', 'wb') as buff:
        pickle.dump({'model': model, 'trace': trace}, buff)

    pymc.traceplot(trace)
    plt.savefig('./plot.png')
