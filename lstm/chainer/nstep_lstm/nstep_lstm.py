#!/usr/bin/env python
# -*- coding: utf-8 -*-

from params import *  # noqa
import chainer.links as L
import chainer.functions as F
import chainer
import numpy as np
import random
# https://qiita.com/chantera/items/d8104012c80e3ea96df7

xp = np
if GPU >= 0:
    xp = chainer.cuda.cupy
    # always run the same calcuation
    np.random.seed(SEED)
    print('use gpu')
else:
    print('use cpu')

# always run the same calcuation
xp.random.seed(SEED)
random.seed(SEED)


class LSTM(L.NStepLSTM):

    def __init__(self, n_layers, in_size, out_size, dropout=0.5, initialW=None, initial_bias=None, **kwargs):
        super(LSTM, self).__init__(n_layers, in_size, out_size, dropout, initialW, initial_bias, **kwargs)
        with self.init_scope():
            self.reset_state()

    def __call__(self, xs):
        hy, cy, ys = super(LSTM, self).__call__(self.hx, self.cx, xs)
        self.hx = hy
        self.cx = cy
        return ys

    def reset_state(self):
        self.hx = None
        self.cx = None

    def to_cpu(self):
        super(LSTM, self).to_cpu()
        if self.cx is not None:
            self.cx.to_cpu()
        if self.hx is not None:
            self.hx.to_cpu()

    def to_gpu(self, device=None):
        super(LSTM, self).to_gpu(device)
        if self.cx is not None:
            self.cx.to_gpu(device)
        if self.hx is not None:
            self.hx.to_gpu(device)


if __name__ == '__main__':
    n_layers = 1
    in_size = 10
    seq_size = 7
    out_size = 5
    batch_size = 3
    gpu = -1

    lstm = LSTM(n_layers, in_size, out_size)
    if GPU >= 0:
        chainer.cuda.get_device(GPU).use()
        lstm.to_gpu(GPU)

    xp = np
    if GPU >= 0:
        xp = chainer.cuda.cupy

    x = xp.arange(seq_size * in_size).reshape(seq_size, in_size).astype(np.float32)
    v = chainer.Variable(x)
    vs = [v] * batch_size
    assert batch_size == len(vs)
    assert (seq_size, in_size) == vs[0].shape
    y = lstm(vs)
    assert list == type(y)
    assert batch_size == len(y)
    assert (seq_size, out_size) == y[0].shape
    z = F.concat(y, axis=0)
    assert (batch_size * seq_size, out_size) == z.shape
    target_size = 4
    linear = L.Linear(out_size, target_size)
    if GPU >= 0:
        linear.to_gpu(GPU)
    s = linear(z)
    assert (batch_size * seq_size, target_size) == s.shape
