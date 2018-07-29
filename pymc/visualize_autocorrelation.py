#!/usr/bin/env python
# -*- coding: utf-8 -*-
from params import *  # noqa
import pymc
# import matplotlib.pyplot as plt
from params import *  # noqa
# import os
from pymc.Matplot import plot as mcplot

if __name__ == '__main__':
    mcmc = pymc.database.pickle.load(PICKLE_PATH)
    mcplot(mcmc.trace('ws'), common_scale=False)
