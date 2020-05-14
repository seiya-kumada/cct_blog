#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch


class QmuUpdater:

    def __init__(self, K, D, hyper_params):
        self.K = K
        self.D = D
        self.m = torch.empty(K, D)
        self.Lambda = torch.empty(K, D, D)
        self.beta = torch.torch.empty(K)
        self.hyper_params = hyper_params

    def update(self, dataset, eta):
        N, _ = dataset.size()
        self.beta = torch.sum(eta, dim=0) + self.hyper_params.beta
        s = 0
        for n in range(N):
            x = dataset[n]
            s += eta[n] * x
        self.m = (s + self.hyper_params.beta * self.hyper_params.m) / self.beta
