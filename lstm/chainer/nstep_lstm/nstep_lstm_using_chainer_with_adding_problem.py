#!/usr/bin/env python
# -*- coding: utf-8 -*-

from params_for_adding_problem import *  # noqa
import chainer
from sklearn.model_selection import train_test_split
from chainer import optimizers
import chainer.links as L
import chainer.functions as F
from chainer import serializers
import _pickle
import nstep_lstm
from make_adding_problem import *  # noqa
import sys
# https://qiita.com/aonotas/items/8e38693fb517e4e90535

xp = np
if GPU >= 0:
    xp = chainer.cuda.cupy


# LSTM
class MyNet(chainer.Chain):

    def __init__(self, n_layers=1, n_in=1, n_hidden=20, n_out=1, dropout=0.5, train=True):
        super(MyNet, self).__init__()
        with self.init_scope():
            self.l1 = nstep_lstm.LSTM(n_layers, n_in, n_hidden, dropout)
            self.l2 = L.Linear(n_hidden, n_out, initialW=chainer.initializers.Normal(scale=0.01))
            self.train = train

    def __call__(self, x):
        with chainer.using_config('train', self.train):
            # x.shape: [(seq_size, n_in)] * batch_size
            h = self.l1(x)  # [(seq_size, n_hidden)] * batch_size
            # assert len(h) == BATCH_SIZE
            assert h[0].shape == (SEQUENCE_SIZE, N_HIDDEN)
            h = [v[-1, :].reshape(1, -1) for v in h]
            # assert len(h) == BATCH_SIZE
            assert h[0].shape == (1, N_HIDDEN)
            h = F.concat(h, axis=0)
            # assert h.shape == (BATCH_SIZE, N_HIDDEN)
            y = self.l2(h)
            # assert y.shape == (BATCH_SIZE, N_OUT)
        return y

    def reset_state(self):
        self.l1.reset_state()


# 損失値計算器
class LossCalculator(chainer.Chain):

    def __init__(self, model):
        super(LossCalculator, self).__init__()
        with self.init_scope():
            self.model = model

    # x.shape: [(seq_size, n_in)] * batch_size
    # t.shape: [(n_out,)] * batch_size
    def __call__(self, x, t):
        y = self.model(x)
        assert y.shape == (BATCH_SIZE, N_OUT)
        t = F.concat([a.reshape(1, -1) for a in t], axis=0)
        assert t.shape == (BATCH_SIZE, N_OUT)
        loss = F.mean_squared_error(y, t)
        return loss


# バッチ単位で1つのシーケンスを学習する。
def calculate_loss(model, x_data, t_data):
    batch_size, seq_size, n_in = x_data.shape
    assert seq_size == SEQUENCE_SIZE
    assert n_in == N_IN
    batch_size, n_out = t_data.shape
    assert n_out == N_OUT
    assert batch_size == BATCH_SIZE

    xs = []
    ts = []
    for x, t in zip(x_data, t_data):
        # assert x.shape == (SEQUENCE_SIZE, N_IN)
        # assert t.shape == (N_OUT,)
        xs.append(chainer.Variable(x.astype(dtype=xp.float32)))
        ts.append(chainer.Variable(t.astype(dtype=xp.float32)))
    loss = model(xs, ts)
    return loss


# モデルを更新する。
def update_model(model, xs, ys):
    # xs.shape == (batch_size, seq_size, n_in)
    # ys.shape == (batch_size, n_out)
    loss = calculate_loss(model, xs, ys)

    # 誤差逆伝播
    loss_calculator.cleargrads()
    loss.backward()

    # バッチ単位で古い記憶を削除し、計算コストを削減する。
    loss.unchain_backward()

    # バッチ単位で更新する。
    optimizer.update()
    return loss


# テストデータに対する誤差を計算する。
def evaluate(loss_calculator, all_xs, all_ys):
    # all_xs.shape == (val_size, seq_size, n_in)
    # all_ys.shape == (val_size, n_out)
    batches = all_xs.shape[0] // BATCH_SIZE
    clone = loss_calculator.copy()
    clone.train = False
    clone.model.reset_state()
    start = 0
    for i in range(batches):
        xs = all_xs[start: start + BATCH_SIZE]
        ys = all_ys[start: start + BATCH_SIZE]
        start += BATCH_SIZE

        loss = calculate_loss(clone, xs, ys)
    return loss


if __name__ == '__main__':

    # _/_/_/ データの作成

    data, target = make_adding_problem(TOTAL_SIZE, SEQUENCE_SIZE)
    if GPU >= 0:
        data = chainer.cuda.to_gpu(data)
        target = chainer.cuda.to_gpu(target)

    assert data.shape == (TOTAL_SIZE, SEQUENCE_SIZE, N_IN)
    assert target.shape == (TOTAL_SIZE, N_OUT)

    # 訓練データと検証データに分ける。
    n_train = int(TOTAL_SIZE * SPRIT_RATE)
    n_val = TOTAL_SIZE - n_train
    train_x, val_x, train_y, val_y = train_test_split(data, target, test_size=n_val)
    assert train_x.shape == (n_train, SEQUENCE_SIZE, N_IN)
    assert train_y.shape == (n_train, N_OUT)
    assert val_x.shape == (n_val, SEQUENCE_SIZE, N_IN)
    assert val_y.shape == (n_val, N_OUT)

    # _/_/_/ モデルの設定

    mynet = MyNet(N_LAYERS, N_IN, N_HIDDEN, N_OUT, DROPOUT)
    if GPU >= 0:
        mynet.to_gpu()

    loss_calculator = LossCalculator(mynet)

    # _/_/_/ 最適化器の作成

    optimizer = optimizers.Adam()
    optimizer.setup(loss_calculator)

    # _/_/_/ 訓練

    batches = train_x.shape[0] // BATCH_SIZE
    print('batches: {}'.format(batches))
    losses = []
    val_losses = []
    for epoch in range(EPOCHS):
        # エポックの最初でシャッフルする。
        indices = np.random.permutation(n_train)
        train_x = train_x[indices]
        train_y = train_y[indices]

        start = 0
        for i in range(batches):
            xs = train_x[start: start + BATCH_SIZE]
            ys = train_y[start: start + BATCH_SIZE]
            start += BATCH_SIZE

            # バッチ単位でモデルを更新する。
            loss = update_model(loss_calculator, xs, ys)

        # 検証する。
        val_loss = evaluate(loss_calculator, val_x, val_y)

        # エポック単位の表示
        average_loss = loss.data
        average_val_loss = val_loss.data
        print('epoch:{}, loss:{}, val_loss:{}'.format(epoch, average_loss, average_val_loss))

        losses.append(average_loss)
        val_losses.append(average_val_loss)

    # 保存する。
    serializers.save_npz('./chainer_mynet_dropout={}.npz'.format(DROPOUT), mynet)
    _pickle.dump(losses, open('./chainer_losses_dropout={}.pkl'.format(DROPOUT), 'wb'))
    _pickle.dump(val_losses, open('./chainer_val_losses_dropout={}.pkl'.format(DROPOUT), 'wb'))
