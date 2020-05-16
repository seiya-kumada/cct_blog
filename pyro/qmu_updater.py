#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch


class QmuUpdater:

    def __init__(self, K, D, hyper_params):
        self.K = K
        self.D = D
        self.m = torch.empty(K, D)
        self.beta = torch.torch.empty(K)
        self.hyper_params = hyper_params

    # eta: (N, K)
    # dataset: (N,D)
    def update(self, dataset, eta):
        N, _ = dataset.size()
        self.beta = torch.matmul(torch.transpose(eta), torch.ones(self.K)) + self.hyper_params.beta
        self.m = (torch.matmul(torch.transpose(eta), dataset) + self.hyper_params.beta * self.hyper_params.m) / self.beta
