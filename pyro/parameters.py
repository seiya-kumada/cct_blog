#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
import unittest


class HyperParameters:

    def __init__(self, dim, k, nu):
        self.beta = 0.1
        self.m = torch.zeros(dim).reshape(1, -1)  # (1,dim)
        self.W = torch.eye(dim)  # (dim,dim)
        self.alpha = torch.ones(k).reshape(k)  # (k)
        if nu <= dim - 1:
            raise ValueError("nu must be greater than dim - 1")
        self.nu = nu


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
