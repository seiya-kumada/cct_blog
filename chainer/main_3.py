#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import chainer
from chainer import Variable
import chainer.functions as F
import chainer.links as L

EPOCHS = 300
DATA_SIZE = 64
INPUT_SIZE = 1000
HIDDEN_SIZE = 100
OUTPUT_SIZE = 10
LEARNING_RATE = 1.0e-04

# set a specified seed to random value generator in order to reproduce the same results
np.random.seed(1)

X = np.random.randn(DATA_SIZE, INPUT_SIZE).astype(np.float32)
Y = np.random.randn(DATA_SIZE, OUTPUT_SIZE).astype(np.float32)
W1 = np.random.randn(INPUT_SIZE, HIDDEN_SIZE).astype(np.float32)
W2 = np.random.randn(HIDDEN_SIZE, OUTPUT_SIZE).astype(np.float32)


class TwoLayerNet(chainer.Chain):

    def __init__(self, d_in, h, d_out):
        super(TwoLayerNet, self).__init__(
            linear1=L.Linear(d_in, h,  initialW=W1.transpose().copy()),
            linear2=L.Linear(h, d_out, initialW=W2.transpose().copy())
        )

    def __call__(self, x):
        g = self.linear1(x)
        h_relu = F.relu(g)
        y_pred = self.linear2(h_relu)
        return y_pred


def sample_3():
    # create random input and output data
    x = Variable(X.copy())
    y = Variable(Y.copy())

    # create a network
    model = TwoLayerNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)

    for t in range(EPOCHS):
        # forward
        y_pred = model(x)

        # compute and print loss
        loss = F.mean_squared_error(y_pred, y)
        print(loss.data)

        # zero the gradients
        model.cleargrads()

        # backward
        loss.backward()

        # update weights
        model.linear1.W.data -= LEARNING_RATE * model.linear1.W.grad
        model.linear2.W.data -= LEARNING_RATE * model.linear2.W.grad


if __name__ == '__main__':
    sample_3()
