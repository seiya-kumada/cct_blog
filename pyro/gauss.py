#!/usr/bin/env python
# -*- coding:utf-8 -*-
import unittest
import torch.distributions as dist
import torch
import math


class Gauss:

    def __init__(self, mu, Lambda):
        self.dist = dist.multivariate_normal.MultivariateNormal(mu, precision_matrix=Lambda)

    def sample(self):
        return self.dist.sample()

    def probs(self, x):
        return torch.exp(self.dist.log_prob(x))


class TestGauss(unittest.TestCase):

    def test(self):
        D = 1
        mu = torch.zeros(D)
        Lambda = torch.eye(D)
        d = Gauss(mu, Lambda)
        s = d.sample()
        self.assertTrue((D,) == s.size())
        p = d.probs(torch.zeros(D))
        self.assertAlmostEqual(p, 1.0 / math.sqrt(2 * math.pi), 1.0e-4)


if __name__ == "__main__":
    unittest.main()
