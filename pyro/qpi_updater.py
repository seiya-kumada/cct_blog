#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
import unittest
import parameters as pa


class QpiUpdater:

    def __init__(self, hyper_params):
        self.alpha = hyper_params.alpha
        self.hyper_params = hyper_params

    def update(self, dataset, eta):
        # eta:(N,K)
        # dataset:(N,D)
        N, K = eta.size()
        self.alpha = torch.matmul(torch.t(eta), torch.ones(N, dtype=torch.float32)) + self.hyper_params.alpha


class TestQpiUpdater(unittest.TestCase):

    def test(self):
        N = 2
        D = 3
        K = 4
        eta = torch.arange(N * K, dtype=torch.float32).reshape(N, K)
        dataset = torch.arange(N * D, dtype=torch.float32).reshape(N, D)
        hyper_params = pa.HyperParameters(dim=D, k=K, nu=D * torch.ones(K))
        updater = QpiUpdater(hyper_params)
        updater.update(dataset, eta)
        self.assertTrue((K,) == updater.alpha.size())


if __name__ == "__main__":
    unittest.main()
