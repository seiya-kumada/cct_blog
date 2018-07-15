#!/usr/bin/env python
# -*- coding: utf-8 -*-
from params import *  # noqa
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    ymeans = np.load(YMEANS_PATH)
    ystds = np.load(YSTDS_PATH)
    ixs = np.load(IXS_PATH)
    ypredictions = np.load(YPREDICTIONS_PATH)

    plt.title('Bayesian Inference')

    # draw a predictive curve
    plt.plot(ixs, ystds, label='predictive std', linestyle='dashed')

    plt.legend(loc='best')
    plt.savefig('results/mcstds_{}.png'.format(M))
