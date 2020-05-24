#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
import unittest


class HyperParameters:

    def __init__(self, dim, k, nu):
        self.beta = 0.1 * torch.ones(k)
        self.m = torch.zeros(k, dim)
        ws = [torch.eye(dim) for _ in range(k)]
        self.W = torch.stack(ws, dim=0)
        self.alpha = torch.ones(k)
        if nu[0] <= dim - 1:
            raise ValueError("nu must be greater than dim - 1")
        self.nu = nu


class TestHyerParameters(unittest.TestCase):

    def test_init(self):
        K = 3
        DIM = 10
        NU = DIM * torch.ones(K)
        hyper_params = HyperParameters(dim=DIM, k=K, nu=NU)
        self.assertAlmostEqual(0.1, hyper_params.beta[0])
        self.assertTrue(torch.all(hyper_params.m[0] == torch.zeros(DIM)))
        self.assertTrue(torch.all(hyper_params.W[0] == 10 * torch.eye(DIM)))
        self.assertTrue(torch.all(hyper_params.alpha == torch.ones(K)))
        self.assertAlmostEqual(hyper_params.nu[0], NU[0])


if __name__ == "__main__":
    unittest.main()
