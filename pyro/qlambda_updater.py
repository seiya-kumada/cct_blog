#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch


class QlambdaUpdater:

    def __init__(self, K, hyper_params):
        self.K = K
        self.hyper_params = hyper_params
        self.nu = None
        self.W = None

    def update(self, eta, beta, m, dataset):
        # eta: (N,K)
        # dataset: (N,D)
        # beta: (K,1)
        # m: (K,D)
        N, _ = dataset.size()
        p = torch.einsum('nd,ne->nde', dataset, dataset)
        a = torch.einsum('kn,nde->kde', torch.transpose(eta), p)  # K,D,D
        b = torch.matmul(self.hyper_params.m, torch.t(self.hyper_params.m))
        c = torch.einsum('kd,ke->kde', m, m)  # K,D,D
        self.W = (a + self.hyper_params.beta * b - beta * c + self.hyper_params.inverse()).inverse()
        self.nu = torch.matmul(torch.t(eta), torch.ones(N)) + self.hyper_params.nu
