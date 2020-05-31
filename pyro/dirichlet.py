#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
import unittest
import parameters as pa
import torch.distributions as dist


class Dirichlet:

    def __init__(self, alpha):
        self.dist = dist.dirichlet.Dirichlet(alpha)

    def sample(self):
        return self.dist.sample()


class TestDirichlet(unittest.TestCase):

    def test(self):
        D = 3
        K = 4
        hyper_params = pa.HyperParameters(dim=D, k=K, nu=D * torch.ones(K))
        d = Dirichlet(hyper_params.alpha)
        s = d.sample()
        self.assertTrue(s.size() == (K,))


if __name__ == "__main__":
    unittest.main()
