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

    def update(self, dataset, eta, beta, m):
        # eta: (N,K)
        # dataset: (N,D)
        # beta: (K)
        # m: (K,D)
        p = torch.einsum('nd,ne->nde', dataset, dataset)
        a = torch.einsum('kn,nde->kde', torch.t(eta), p)  # K,D,D
        K, D = m.size()
        b = torch.einsum('kd,ke->kde', self.hyper_params.m, self.hyper_params.m)  # K,D,D
        c = torch.einsum('kd,ke->kde', m, m)  # K,D,D
        self.W = (a + self.hyper_params.beta.reshape(-1, 1, 1) * b - beta.reshape(K, 1, 1) * c + self.hyper_params.W.inverse()).inverse()

        min_det = torch.min(torch.det(self.W))
        if min_det < 0:
            raise Exception("invalid determinant detected")

        self.hyper_params.beta.reshape(-1, 1, 1) * b
        N, _ = dataset.size()
        self.nu = torch.matmul(torch.t(eta), torch.ones(N, dtype=torch.float32)) + self.hyper_params.nu


class TestQlambdaUpdater(unittest.TestCase):

    def test(self):
        N = 2
        D = 3
        K = 4
        eta = torch.arange(N * K, dtype=torch.float32).reshape(N, K)
        beta = torch.arange(K, dtype=torch.float32)
        m = torch.arange(K * D, dtype=torch.float32).reshape(K, D)
        dataset = torch.arange(N * D, dtype=torch.float32).reshape(N, D)
        hyper_params = pa.HyperParameters(dim=D, k=K, nu=D * torch.ones(K))
        updater = QlambdaUpdater(hyper_params)
        updater.update(dataset, eta, beta, m)
        self.assertTrue((K, D, D) == updater.W.size())
        self.assertTrue((K,) == updater.nu.size())


if __name__ == "__main__":
    unittest.main()
