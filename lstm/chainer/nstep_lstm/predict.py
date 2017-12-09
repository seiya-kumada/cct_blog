#!/usr/bin/env python
# -*- coding: utf-8 -*-

from nstep_lstm_using_chainer_with_fibonacci import MyNet, DatasetMaker
from chainer import serializers
import chainer
import numpy as np
import matplotlib.pyplot as plt
import _pickle

VALUE = 5
N_IN = 1
N_HIDDEN = 200
N_OUT = 1
TOTAL_SIZE = 2000
SEQUENCE_SIZE = 50
PLOT_SIZE = 4 * SEQUENCE_SIZE


def predict_seq(model, input_seq):
    seq_size = len(input_seq)
    assert seq_size == SEQUENCE_SIZE
    for i in range(seq_size):
        x = chainer.Variable(np.asarray(input_seq[i:i + 1], dtype=np.float32)[:, np.newaxis])
        y = model([x])
    return y[0].data


def predict(model, dataset, seq_size):
    input_seq = dataset[:seq_size].copy()
    assert input_seq.shape == (seq_size,)
    output_seq = np.zeros(seq_size)

    assert len(dataset) == TOTAL_SIZE
    model.train = False
    model.reset_state()
    for i in range(len(dataset) - seq_size):
        y = predict_seq(model, input_seq)

        # 先頭の要素を削除する。
        input_seq = np.delete(input_seq, 0)

        # 末尾にいま予測した値を追加する。
        input_seq = np.append(input_seq, y)

        # 予測値を保存する。
        output_seq = np.append(output_seq, y)

        if i == 200:
            break
    return output_seq


if __name__ == '__main__':

    # _/_/_/ モデルの読み込み

    mynet = MyNet(N_IN, N_HIDDEN, N_OUT)
    serializers.load_npz('chainer_mynet.npz', mynet)

    # _/_/_/ データの作成

    dataset = DatasetMaker.make(TOTAL_SIZE, VALUE)

    # _/_/_/ 予測

    output_seq = predict(mynet, dataset, SEQUENCE_SIZE)

    # _/_/_/ 視覚化

    # 予測した時系列データ
    plt.figure(figsize=(10, 5))
    plt.xlim([0, PLOT_SIZE])
    plt.plot(dataset, linestyle='dotted', color='red')
    plt.plot(output_seq, color='black')
    plt.show()

    # 誤差とエポックの間の関係
    losses = _pickle.load(open('./chainer_losses.pkl', 'rb'))
    val_losses = _pickle.load(open('./chainer_val_losses.pkl', 'rb'))
    plt.plot(losses, label='loss')
    plt.plot(val_losses, label='val_loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
