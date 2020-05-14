#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
import unittest


# class Parameters:
#
#     def __init__(self, dim, k):
#         self.mu = torch.zeros(k, dim)
#         tmp = [torch.eye(dim) for _ in range(k)]
#         self.Lambda = torch.stack(tmp, dim=0)
#         self.eta = torch.ones(k) / k


class HyperParameters:

    def __init__(self, dim, k, nu):
        self.beta = 0.0
        self.m = torch.zeros(dim)
        self.W = torch.eye(dim)
        self.alpha = torch.ones(k)
        if nu <= dim - 1:
            raise ValueError("nu must be greater than dim - 1")
        self.nu = nu


# class TestParameters(unittest.TestCase):

    # def test_init(self):
    #     K = 3
    #     DIM = 10
    #     params = Parameters(dim=DIM, k=K)
    #     self.assertTrue(torch.all(params.mu == torch.zeros(K, DIM)))
    #     self.assertTrue(params.Lambda.size() == (K, DIM, DIM))
    #     self.assertEqual(1, torch.sum(params.eta))

class TestHyerParameters(unittest.TestCase):

    def test_init(self):
        K = 3
        DIM = 10
        NU = DIM
        hyper_params = HyperParameters(dim=DIM, k=K, nu=NU)
        self.assertAlmostEqual(0.0, hyper_params.beta)
        self.assertTrue(torch.all(hyper_params.m == torch.zeros(DIM)))
        self.assertTrue(torch.all(hyper_params.W == torch.eye(DIM)))
        self.assertTrue(torch.all(hyper_params.alpha == torch.ones(K)))
        self.assertAlmostEqual(hyper_params.nu, NU)


if __name__ == "__main__":
    unittest.main()
