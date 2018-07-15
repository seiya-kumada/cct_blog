#!/usr/bin/env python
# -*- coding: utf-8 -*-
from params import *  # noqa
import pymc
import matplotlib.pyplot as plt
from params import *  # noqa
import os


def plot_posterior(mcmc, index):
    name = 'w{}'.format(index)
    samples = mcmc.trace(name)[:]
    plt.figure(figsize=(12, 6))
    label = r'$p(w_{}|D)$'.format(index)
    plt.hist(samples, histtype='stepfilled', bins=50, normed=True, label=label)
    plt.ylabel(label, fontsize=20)
    plt.xlabel(r'$w_{}$'.format(index), fontsize=20)
    plt.legend(loc='best')
    plt.savefig(os.path.join('./results', '{}.png'.format(name)))
    plt.show()


if __name__ == '__main__':
    mcmc = pymc.database.pickle.load(PICKLE_NAME)
    for i in range(M):
        plot_posterior(mcmc, i)
