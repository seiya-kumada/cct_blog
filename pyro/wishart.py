#!/usr/bin/env python
# -*- coding:utf-8 -*-
import unittest
import scipy.stats as stats
import numpy as np


class Wishart:

    # nu:(), W:(D,D)
    def __init__(self, nu, W):
        self.dist = stats.wishart(nu, W)

    def sample(self):
        return self.dist.rvs()


class TestGauss(unittest.TestCase):

    def test(self):
        D = 2
        nu = 3
        W = np.eye(D)
        w = Wishart(nu, W)
        a = w.sample()
        self.assertTrue(a.shape == (D, D))


if __name__ == "__main__":
    unittest.main()
