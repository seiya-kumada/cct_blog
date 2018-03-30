#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import chainer
import chainer.functions as F
import chainer.optimizers as P
import chainer.links as L
import chainer.datasets as D
import chainer.iterators as Iter
from chainer import initializers
from chainer import training
from chainer.training import extensions
from params import *  # noqa


# set a specified seed to random value generator in order to reproduce the same results
np.random.seed(1)


class MyNet(chainer.Chain):

    def __init__(self, length_scale, d_in, h, d_out, dropout_ratio=0.5, train=True):

        W = initializers.HeNormal(1 / length_scale)
        bias = initializers.Zero()
        super(MyNet, self).__init__()
        with self.init_scope():
            self.linear1 = L.Linear(d_in, h,  initialW=W, initial_bias=bias)
            self.linear2 = L.Linear(h, h, initialW=W, initial_bias=bias)
            self.linear3 = L.Linear(h, h, initialW=W, initial_bias=bias)
            self.linear4 = L.Linear(h, d_out, initialW=W, initial_bias=bias)
            self.train = train
            self.dropout_ratio = dropout_ratio

    def __call__(self, x):
        h = F.dropout(x, ratio=self.dropout_ratio)
        h = self.linear1(x)
        h = F.relu(h)

        h = F.dropout(h, ratio=self.dropout_ratio)
        h = self.linear2(h)
        h = F.relu(h)

        h = F.dropout(h, ratio=self.dropout_ratio)
        h = self.linear3(h)
        h = F.relu(h)

        h = F.dropout(h, ratio=self.dropout_ratio)
        h = self.linear4(h)

        return h


class LossCalculator(chainer.Chain):

    def __init__(self, model):
        super(LossCalculator, self).__init__()
        with self.init_scope():
            self.model = model

    def __call__(self, x, y):
        y_pred = self.model(x)
        loss = F.mean_squared_error(y_pred, y)
        chainer.report({'loss': loss}, self)
        return loss


def train():
    # _/_/_/ load dataset

    xs = np.load(XS_PATH)
    ys = np.load(YS_PATH)

    # _/_/_/ split dataset and make iterators

    # for training
    train_dataset = D.TupleDataset(xs, ys)
    train_iter = Iter.SerialIterator(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # _/_/_/ create a network

    model = MyNet(LENGTH_SCALE, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, DROPOUT_RATIO)
    loss_calculator = LossCalculator(model)

    # _/_/_/ create an optimizer

    # optimizer = P.SGD(lr=LEARNING_RATE)
    optimizer = P.Adam()
    # ptimizer = P.RMSprop(lr=LEARNING_RATE)

    # _/_/_/ connect the optimizer with the network

    optimizer.setup(loss_calculator)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=WEIGHT_DECAY))

    # _/_/_/ make a updater

    updater = training.StandardUpdater(train_iter, optimizer)

    # _/_/_/ make a trainer

    epoch_interval = (1, 'epoch')
    model_interval = (EPOCHS, 'epoch')

    trainer = training.Trainer(updater, (EPOCHS, 'epoch'), out=OUTPUT_DIR_PATH)
    # trainer.extend(extensions.ExponentialShift('lr', 0.99), trigger=epoch_interval)

    # save a trainer
    trainer.extend(extensions.snapshot(), trigger=model_interval)

    # save a model
    trainer.extend(extensions.snapshot_object(model, MODEL_NAME), trigger=model_interval)

    trainer.extend(extensions.LogReport(trigger=epoch_interval))
    trainer.extend(
        extensions.PrintReport(
            ['epoch', 'iteration', 'main/loss']
        ),
        trigger=epoch_interval)
    trainer.extend(
        extensions.PlotReport(
            ['main/loss'],
            'epoch',
            file_name='loss.png'
        )
    )
    # _/_/_/ run

    trainer.run()


if __name__ == '__main__':
    train()
