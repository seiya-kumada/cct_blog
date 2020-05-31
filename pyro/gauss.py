#!/usr/bin/env python
# -*- coding:utf-8 -*-
import unittest
import torch.distributions as dist
import torch


class Gauss:

    def __init__(self, mu, Lambda):
        self.dist = dist.multivariate_normal.MultivariateNormal(mu, precision_matrix=Lambda)

    def sample(self):
        return self.dist.sample()


class TestGauss(unittest.TestCase):

    def test(self):
        D = 3
        mu = torch.ones(D)
        Lambda = torch.eye(D)
        d = Gauss(mu, Lambda)
        s = d.sample()
        self.assertTrue((D,) == s.size())


if __name__ == "__main__":
    unittest.main()
