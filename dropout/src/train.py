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
from params import *  # noqa


# set a specified seed to random value generator in order to reproduce the same results
np.random.seed(1)

# initial weights
W1 = np.random.randn(INPUT_SIZE, HIDDEN_SIZE).astype(np.float32)
W2 = np.random.randn(HIDDEN_SIZE, OUTPUT_SIZE).astype(np.float32)


class MyNet(chainer.Chain):

    def __init__(self, d_in, h, d_out, dropout_ratio=0.5, train=True):
        super(MyNet, self).__init__()
        with self.init_scope():
            self.linear1 = L.Linear(d_in, h,  initialW=W1.transpose().copy())
            self.linear2 = L.Linear(h, d_out, initialW=W2.transpose().copy())
            self.train = train
            self.dropout_ratio = dropout_ratio

    def __call__(self, x):
        h = F.dropout(x, ratio=self.dropout_ratio)
        h = self.linear1(x)
        h = F.relu(h)
        h = F.dropout(h, ratio=self.dropout_ratio)
        h = self.linear2(h)
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
    n_train = int(SPLIT_RATE * SAMPLE_SIZE)
    train_dataset = D.TupleDataset(xs[: n_train], ys[: n_train])
    train_iter = Iter.SerialIterator(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # for test
    test_dataset = D.TupleDataset(xs[n_train:], ys[n_train:])
    test_iter = Iter.SerialIterator(test_dataset, batch_size=BATCH_SIZE, repeat=False, shuffle=False)

    # _/_/_/ create a network

    model = MyNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, DROPOUT_RATIO)
    loss_calculator = LossCalculator(model)

    # _/_/_/ create an optimizer

    # optimizer = P.SGD(lr=LEARNING_RATE)
    optimizer = P.Adam()

    # _/_/_/ connect the optimizer with the network

    optimizer.setup(loss_calculator)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=WEIGHT_DECAY))

    # _/_/_/ make a updater

    updater = training.StandardUpdater(train_iter, optimizer)

    # _/_/_/ make a trainer

    epoch_interval = (1, 'epoch')
    model_interval = (EPOCHS, 'epoch')

    trainer = training.Trainer(updater, (EPOCHS, 'epoch'), out=OUTPUT_DIR_PATH)

    # save a trainer
    trainer.extend(extensions.snapshot(), trigger=model_interval)

    # save a model
    trainer.extend(extensions.snapshot_object(model, MODEL_NAME), trigger=model_interval)

    trainer.extend(extensions.LogReport(trigger=epoch_interval))
    trainer.extend(extensions.Evaluator(test_iter, loss_calculator), trigger=epoch_interval)
    trainer.extend(
        extensions.PrintReport(
            ['epoch', 'iteration', 'main/loss', 'validation/main/loss']
        ),
        trigger=epoch_interval)
    trainer.extend(
        extensions.PlotReport(
            ['main/loss', 'validation/main/loss'],
            'epoch',
            file_name='loss.png'
        )
    )
    # _/_/_/ run

    trainer.run()


if __name__ == '__main__':
    train()
