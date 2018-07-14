#!/usr/bin/env python
# -*- coding: utf-8 -*-
from params import *  # noqa
import pymc
from params import *  # noqa
import numpy as np


def linear_regression(xs, ws):
    return xs.dot(ws)


if __name__ == '__main__':
    # load a model
    mcmc = pymc.database.pickle.load(PICKLE_NAME)

    # extract samples for posteriors
    ws = np.empty((M, (ITER - BURN) // THIN), dtype=np.float64)
    for i in range(M):
        ws[i] = mcmc.trace('w{}'.format(i))[:]

    # calculate x^0,x^1,...,x^{M-1}
    ixs = np.linspace(XMIN, XMAX, XCOUNT)
    xs = np.empty((XCOUNT, M))
    for i in range(M):
        xs[:, i] = np.power(ixs, i)

    # calculate ys
    ys = linear_regression(xs, ws)

    # calculate statistics
    ymeans = ys.mean(axis=1)
    ystds = ys.std(axis=1)

    # save them
    np.save('ymeans.npy', ymeans)
    np.save('ystds.npy', ystds)
