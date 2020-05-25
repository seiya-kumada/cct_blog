#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
import unittest
import parameters as pa


class QmuUpdater:

    def __init__(self, hyper_params):
        self.m = torch.tensor([[-0.1, 0.1], [0.1, 0.2], [0.15, 0.1]], dtype=torch.float32)
        self.beta = hyper_params.beta
        self.hyper_params = hyper_params

    # eta: (N, K)
    # dataset: (N,D)
    def update(self, dataset, eta):
        N, _ = eta.size()
        self.beta = torch.matmul(torch.t(eta), torch.ones(N, dtype=torch.float32)) + self.hyper_params.beta
        self.m = (torch.einsum("kn,nd->kd", torch.t(eta), dataset)
                  + self.hyper_params.beta.reshape(-1, 1) * self.hyper_params.m) / self.beta.reshape(-1, 1)


class TestQmuUpdater(unittest.TestCase):

    def test(self):
        K = 3
        D = 2
        N = 4
        dataset = torch.arange(N * D, dtype=torch.float32).reshape(N, D)
        eta = torch.arange(N * K, dtype=torch.float32).reshape(N, K)
        hyper_params = pa.HyperParameters(dim=D, k=K, nu=D * torch.ones(K))
        updater = QmuUpdater(hyper_params)
        updater.update(dataset, eta)
        self.assertTrue(updater.beta.size() == (K,))
        self.assertTrue(updater.m.size() == (K, D))


if __name__ == "__main__":
    unittest.main()
