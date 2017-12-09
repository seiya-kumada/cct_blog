#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import chainer
import chainer.functions as F
import chainer.optimizers as P
import chainer.links as L
import chainer.datasets as D
import chainer.iterators as Iter
from chainer import training
from chainer.training import extensions
from chainer import reporter

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


class LossCalculator(chainer.Chain):

    def __init__(self, model):
        super(LossCalculator, self).__init__()
        with self.init_scope():
            self.model = model

    def __call__(self, x, y):
        y_pred = self.model(x)
        loss = F.mean_squared_error(y_pred, y)
        reporter.report({'loss': loss}, self)
        return loss


def sample_5():
    # make a iterator
    x = X.copy()
    y = Y.copy()
    dataset = D.TupleDataset(x, y)
    train_iter = Iter.SerialIterator(dataset, batch_size=DATA_SIZE, shuffle=False)

    # create a network
    model = TwoLayerNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    loss_calculator = LossCalculator(model)

    # create an optimizer
    optimizer = P.SGD(lr=LEARNING_RATE)

    # connect the optimizer with the network
    optimizer.setup(loss_calculator)

    # make a updater
    updater = training.StandardUpdater(train_iter, optimizer)

    # make a trainer
    trainer = training.Trainer(updater, (EPOCHS, 'epoch'), out='result')
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'elapsed_time']))

    trainer.run()


if __name__ == '__main__':
    sample_5()
