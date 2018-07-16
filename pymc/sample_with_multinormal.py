#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import utils
import pymc
from params import *  # noqa


if __name__ == '__main__':

    # load a dataset
    dataset = utils.load_dataset(DATASET_PATH)
    observed_xs = dataset[:, 0]
    observed_ys = dataset[:, 1]

    # plot_ground_truth()

    # define a prior for w, which is a multivariable gaussian
    ws = pymc.MvNormal('ws', mu=np.zeros(M), tau=ALPHA * np.eye(M), value=np.zeros(M))

    # calculate x^0, x^1, ..., x^{M-1}
    xs = np.empty((M, len(observed_xs)), dtype=np.float32)
    for i in range(M):
        xs[i] = np.power(observed_xs, i)
    assert(xs.shape == (M, len(observed_xs)))

    # define a deterministic function
    linear_regression = pymc.Lambda('linear_regression', lambda xs=xs, ws=ws: ws.dot(xs))

    # define a model likelihood
    y = pymc.Normal('y', mu=linear_regression, tau=TAU, value=observed_ys, observed=True)

    # make a model
    model = pymc.Model([y, ws])
    map_ = pymc.MAP(model)
    map_.fit()
    mcmc = pymc.MCMC(model, db='pickle', dbname=PICKLE_PATH)

    # sampling
    mcmc.sample(iter=ITER, burn=BURN, thin=THIN)
    mcmc.db.close()

    # save it as csv
    mcmc.write_csv(CSV_PATH, variables=['ws'])
