#!/usr/bin/env python
# -*- coding:utf-8 -*-
import unittest
import scipy.stats as stats
import numpy as np


class Wishart:

    # nu:(K,), W:(K,D,D)
    def __init__(self, nu, W):
        self.dists = []
        for n, w in zip(nu, W):
            self.dists.append(stats.wishart(n, w))

    def sample(self):
        return np.array([d.rvs() for d in self.dists])


class TestGauss(unittest.TestCase):

    def test(self):
        K = 3
        D = 2
        nu = 3 * np.ones((K,))
        ws = [np.eye(D) for _ in range(K)]
        W = np.stack(ws, axis=0)
        w = Wishart(nu, W)
        a = w.sample()
        self.assertTrue(a.shape == (K, D, D))


if __name__ == "__main__":
    unittest.main()
