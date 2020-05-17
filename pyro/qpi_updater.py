#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch


class QpiUpdater:

    def __init__(self, hyper_params):
        self.alpha = None
        self.hyper_params = hyper_params

    def update(self, eta, dataset):
        # eta:(N,K)
        # dataset:(N,D)
        N, _ = eta.size()
        self.alpha = torch.matmul(torch.t(eta), torch.ones(N)) + self.hyper_params.alpha
