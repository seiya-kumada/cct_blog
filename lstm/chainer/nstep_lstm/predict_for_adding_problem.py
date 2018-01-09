#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
from params_for_adding_problem import *  # noqa
from make_adding_problem import *  # noqa

xp = np
if GPU >= 0:
    xp = chainer.cuda.cupy


def predict(model, x_data):
    # xs.shape == (data_size, SEQUENCE_SIZE, N_IN)
    # ys.shape == (data_size, N_OUT)
    xs = []
    for x in x_data:
        xs.append(chainer.Variable(x.astype(dtype=xp.float32)))
    return model(xs)
