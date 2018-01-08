#!/usr/bin/env python
# -*- coding: utf-8 -*-

from nstep_lstm_using_chainer_with_adding_problem import MyNet
from sklearn.model_selection import train_test_split
from chainer import serializers
import chainer
# import numpy as np
# import matplotlib.pyplot as plt
# import _pickle
from params_for_adding_problem import *  # noqa
from make_adding_problem import *  # noqa

# PLOT_SIZE = 4 * SEQUENCE_SIZE
xp = np
if GPU >= 0:
    xp = chainer.cuda.cupy


# def predict_seq(model, input_seq):
#     seq_size = len(input_seq)
#     assert seq_size == SEQUENCE_SIZE
#     for i in range(seq_size):
#         x = chainer.Variable(np.asarray(input_seq[i:i + 1], dtype=np.float32)[:, np.newaxis])
#         y = model([x])
#     return y[0].data
#
#
def predict(model, x_data):
    # xs.shape == (data_size, SEQUENCE_SIZE, N_IN)
    # ys.shape == (data_size, N_OUT)
    xs = []
    for x in x_data:
        xs.append(chainer.Variable(x.astype(dtype=xp.float32)))
    return model(xs)


if __name__ == '__main__':

    # _/_/_/ モデルの読み込み

    mynet = MyNet(N_LAYERS, N_IN, N_HIDDEN, N_OUT)
    serializers.load_npz('chainer_mynet_dropout={}.npz'.format(DROPOUT), mynet)
    if GPU >= 0:
        mynet.to_gpu()

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
    print(val_y[0])

    # _/_/_/ 予測

    ys = predict(mynet, val_x)

    # # _/_/_/ 視覚化

    # # 予測した時系列データ
    # plt.figure(figsize=(10, 5))
    # plt.xlim([0, PLOT_SIZE])
    # plt.plot(dataset, linestyle='dotted', color='red')
    # plt.plot(output_seq, color='black')
    # plt.savefig('/Users/kumada/Documents/cct_blog/nstep_lstm/pred_{}_layers={}_dropout={}.png'.format(
    #     SEQUENCE_SIZE, N_LAYERS, DROPOUT))
    # plt.show()

    # # 誤差とエポックの間の関係
    # losses = _pickle.load(open('./chainer_losses_dropout={}.pkl'.format(DROPOUT), 'rb'))
    # val_losses = _pickle.load(open('./chainer_val_losses_dropout={}.pkl'.format(DROPOUT), 'rb'))
    # plt.plot(losses, label='loss')
    # plt.plot(val_losses, label='val_loss')
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.legend()
    # plt.savefig('/Users/kumada/Documents/cct_blog/nstep_lstm/loss_{}_layers={}_dropout={}.png'.format(
    #     SEQUENCE_SIZE, N_LAYERS, DROPOUT))
    # plt.show()
