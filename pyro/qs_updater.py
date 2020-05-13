#!/usr/bin/env python
# -*- coding:utf-8 -*-
import utils
import unittest
import torch


class QsUpdater:

    def __init__(self, pi, K):
        self.pi = pi
        self.K = K

    def update(self, W, nu, m, beta, alpha, dataset):
        N, _ = dataset.size()
        r = torch.empty(N, self.K)
        for n in range(N):
            for k in range(self.K):
                Wk = W[k]
                nk = nu[k]
                Lambda = utils.update_with_4_119(Wk, nk)
                x = dataset[n]
                a = torch.dot(x, torch.matmul(Lambda, x))
                mk = m[k]
                Lambda_mu = utils.update_with_4_121(Wk, nk, mk)
                b = x.dot(Lambda_mu)
                bk = beta[k]
                c = utils.update_with_122(Wk, nk, mk, bk)
                d = utils.update_with_120(Wk, nk)
                alpha_k = alpha[k]
                e = utils.update_with_4_62(alpha_k, k)
                f = -0.5 * a + b - 0.5 * c + 0.5 * d + e
                r[n, k] = torch.exp(f)

        factors = torch.sum(r, dim=1)
        for n in range(N):
            r[n] = r[n] / factors[n]
        self.pi = r


class TestQsUpdater(unittest.TestCase):

    def test_(self):
        dataset = torch.tensor([[2.0, 1.0], [3.0, 2.0]])
        K = 3
        pi = torch.empty(2, K)
        updater = QsUpdater(pi, K=K)
        W = torch.empty(K, 2, 2)
        nu = torch.empty(K, 2)
        m = torch.empty(K, 2)
        beta = torch.emtpy(K)
        alpha = torch.empty(K)
        updater.update(W, nu, m, beta, alpha, dataset)


if __name__ == "__main__":
    unittest.main()
