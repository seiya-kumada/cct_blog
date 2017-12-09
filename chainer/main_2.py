#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import chainer.functions as F
from chainer import Variable

EPOCHS = 300
DATA_SIZE = 64
N_I = 1000
HIDDEN_SIZE = 100
N_O = 10
LEARNING_RATE = 1.0e-04

# set a specified seed to random value generator in order to reproduce the same results
np.random.seed(1)


def sample_2():
    # create random input and output data
    x = Variable(np.random.randn(DATA_SIZE, N_I).astype(np.float32))
    y = Variable(np.random.randn(DATA_SIZE, N_O).astype(np.float32))

    # randomly initialize weights
    w1 = Variable(np.random.randn(N_I, HIDDEN_SIZE).astype(np.float32))
    w2 = Variable(np.random.randn(HIDDEN_SIZE, N_O).astype(np.float32))

    for t in range(EPOCHS):
        # forward pass: compute predicted y
        h = F.matmul(x, w1)
        h_r = F.relu(h)
        y_p = F.matmul(h_r, w2)

        # compute and print loss
        loss = F.mean_squared_error(y_p, y)
        print(loss.data)

        # manually zero the gradients
        w1.zerograd()
        w2.zerograd()

        # backward pass
        # loss.grad = np.ones(loss.shape, dtype=np.float32)
        loss.backward()

        # update weights
        w1.data -= LEARNING_RATE * w1.grad
        w2.data -= LEARNING_RATE * w2.grad


if __name__ == '__main__':
    sample_2()
