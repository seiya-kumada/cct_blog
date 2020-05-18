#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
import unittest
import parameters as pa


class QlambdaUpdater:

    def __init__(self, hyper_params):
        self.hyper_params = hyper_params
        self.nu = hyper_params.nu
        self.W = hyper_params.W

    def update(self, eta, beta, m, dataset):
        # eta: (N,K)
        # dataset: (N,D)
        # beta: (K,1)
        # m: (K,D)
        p = torch.einsum('nd,ne->nde', dataset, dataset)
        a = torch.einsum('kn,nde->kde', torch.t(eta), p)  # K,D,D
        K, D = m.size()
        b = torch.matmul(self.hyper_params.m, torch.t(self.hyper_params.m))  # D,D
        c = torch.einsum('kd,ke->kde', m, m)  # K,D,D
        self.W = (a + self.hyper_params.beta * b - beta.reshape(K, 1, 1) * c + self.hyper_params.W.inverse()).inverse()
        N, _ = dataset.size()
        nu = torch.matmul(torch.t(eta), torch.ones(N, dtype=float)) + self.hyper_params.nu
        self.nu = nu.reshape(K, 1)


class TestQlambdaUpdater(unittest.TestCase):

    def test(self):
        N = 2
        D = 3
        K = 4
        eta = torch.arange(N * K, dtype=float).reshape(N, K)
        beta = torch.arange(K, dtype=float).reshape(K, 1)
        m = torch.arange(K * D, dtype=float).reshape(K, D)
        dataset = torch.arange(N * D, dtype=float).reshape(N, D)
        hyper_params = pa.HyperParameters(dim=D, k=K, nu=D)
        updater = QlambdaUpdater(hyper_params)
        updater.update(eta, beta, m, dataset)
        self.assertTrue((K, D, D) == updater.W.size())
        self.assertTrue((K, 1) == updater.nu.size())


if __name__ == "__main__":
    unittest.main()
