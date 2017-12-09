#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import chainer
from chainer import Variable
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


def sample_1():
    # create random input and output data
    x = X.copy()
    y = Y.copy()

    # randomly initialize weights
    w1 = W1.copy()
    w2 = W2.copy()

    y_size = np.float32(DATA_SIZE * OUTPUT_SIZE)
    for t in range(EPOCHS):
        # forward pass
        h = x.dot(w1)
        h_relu = np.maximum(h, 0)
        y_pred = h_relu.dot(w2)

        # compute mean squared error and print loss
        loss = np.square(y_pred - y).sum() / y_size
        print(loss)

        # backward pass: compute gradients of loss with respect to w2
        grad_y_pred = 2.0 * (y_pred - y) / y_size
        grad_w2 = h_relu.T.dot(grad_y_pred)

        # backward pass: compute gradients of loss with respect to w1
        grad_h_relu = grad_y_pred.dot(w2.T)
        grad_h = grad_h_relu.copy()
        grad_h[h < 0] = 0
        grad_w1 = x.T.dot(grad_h)

        # update weights
        w1 -= LEARNING_RATE * grad_w1
        w2 -= LEARNING_RATE * grad_w2


def sample_2():
    # create random input and output data
    x = Variable(X.copy())
    y = Variable(Y.copy())

    # randomly initialize weights
    w1 = Variable(W1.copy())
    w2 = Variable(W2.copy())

    for t in range(EPOCHS):
        # forward pass: compute predicted y
        h = F.matmul(x, w1)
        h_relu = F.relu(h)
        y_pred = F.matmul(h_relu, w2)

        # compute and print loss
        loss = F.mean_squared_error(y_pred, y)
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


def sample_4():
    # create random input and output data
    x = Variable(X.copy())
    y = Variable(Y.copy())

    # create a network
    model = TwoLayerNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)

    # create an optimizer
    optimizer = P.SGD(lr=LEARNING_RATE)

    # connect the optimizer with the network
    optimizer.setup(model)

    for t in range(EPOCHS):
        # forward pass: compute predicted y
        y_pred = model(x)

        # compute and print loss
        loss = F.mean_squared_error(y_pred, y)
        print(loss.data)

        # zero the gradients
        model.cleargrads()

        # backward
        loss.backward()

        # update weights
        optimizer.update()


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
