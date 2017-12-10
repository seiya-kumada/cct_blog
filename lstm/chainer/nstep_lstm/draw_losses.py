#!/usr/bin/env python
# -*- coding: utf-8 -*-

from params import *  # noqa
import matplotlib.pyplot as plt
import _pickle


if __name__ == '__main__':

    losses_3 = _pickle.load(open('./chainer_losses_dropout=0.3.pkl', 'rb'))
    losses_4 = _pickle.load(open('./chainer_losses_dropout=0.4.pkl', 'rb'))
    losses_5 = _pickle.load(open('./chainer_losses_dropout=0.5.pkl', 'rb'))
    plt.plot(losses_3, label='loss with dropput=0.3')
    plt.plot(losses_4, label='loss with dropout=0.4')
    plt.plot(losses_5, label='loss with dropout=0.5')
    plt.title('train loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('/Users/kumada/Documents/cct_blog/nstep_lstm/losses_layer=1.png')
    plt.show()

    val_losses_3 = _pickle.load(open('./chainer_val_losses_dropout=0.3.pkl', 'rb'))
    val_losses_4 = _pickle.load(open('./chainer_val_losses_dropout=0.4.pkl', 'rb'))
    val_losses_5 = _pickle.load(open('./chainer_val_losses_dropout=0.5.pkl', 'rb'))
    plt.plot(val_losses_3, label='val_loss with dropput=0.3')
    plt.plot(val_losses_4, label='val_loss with dropout=0.4')
    plt.plot(val_losses_5, label='val_loss with dropout=0.5')
    plt.title('val loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('/Users/kumada/Documents/cct_blog/nstep_lstm/val_losses_layer=1.png')
    plt.show()
