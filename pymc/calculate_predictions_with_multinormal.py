#!/usr/bin/env python
# -*- coding: utf-8 -*-
from params import *  # noqa
import pymc
from params import *  # noqa
import numpy as np


def linear_regression(xs, ws):
    return ws.dot(xs)


if __name__ == '__main__':
    # load a model
    mcmc = pymc.database.pickle.load(PICKLE_PATH)

    # extract samples for posteriors
    ws = mcmc.trace('ws')[:]

    # calculate x^0,x^1,...,x^{M-1}
    ixs = np.linspace(XMIN, XMAX, XCOUNT)
    xs = np.empty((M, XCOUNT))
    for i in range(M):
        xs[i, :] = np.power(ixs, i)

    # calculate ys
    ys = linear_regression(xs, ws)

    # calculate statistics
    ymeans = ys.mean(axis=0)
    ystds = ys.std(axis=0)

    # save them
    np.save(YMEANS_PATH, ymeans)
    np.save(YPREDICTIONS_PATH, ys)
    np.save(YSTDS_PATH, ystds)
    np.save(IXS_PATH, ixs)
