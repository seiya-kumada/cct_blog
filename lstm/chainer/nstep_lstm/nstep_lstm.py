#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer.links as L
import chainer.functions as F
import chainer
import numpy as np


class LSTM(L.NStepLSTM):

    def __init__(self, n_layers, in_size, out_size, dropout=0.5, initialW=None, initial_bias=None, **kwargs):
        super(LSTM, self).__init__(n_layers, in_size, out_size, dropout, initialW, initial_bias, **kwargs)
        with self.init_scope():
            self.reset_state()

    def __call__(self, xs):
        batch = len(xs)
        if self.hx is None:
            self.hx = chainer.Variable(np.zeros((self.n_layers, batch, self.out_size), dtype=xs[0].dtype))
        if self.cx is None:
            self.cx = chainer.Variable(np.zeros((self.n_layers, batch, self.out_size), dtype=xs[0].dtype))
        hy, cy, ys = super(LSTM, self).__call__(self.hx, self.cx, xs)
        self.hx = hy
        self.cx = cy
        return ys

    def reset_state(self):
        self.hx = None
        self.cx = None


if __name__ == '__main__':
    n_layers = 1
    in_size = 10
    seq_size = 7
    out_size = 5
    batch_size = 3
    lstm = LSTM(n_layers, in_size, out_size)
    x = np.arange(seq_size * in_size).reshape(seq_size, in_size).astype(np.float32)
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
    s = linear(z)
    assert (batch_size * seq_size, target_size) == s.shape
