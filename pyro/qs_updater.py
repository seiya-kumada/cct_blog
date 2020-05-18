#!/usr/bin/env python
# -*- coding:utf-8 -*-
import utils
import unittest
import torch


class QsUpdater:

    def __init__(self):
        self.eta = None

    # W:(K,D,D), nu:(K,1), m:(K,D), beta:(K,1), alpha:(K,1)
    def update(self, dataset, W, nu, m, beta, alpha):
        Phi = torch.einsum('nd,ne->nde', dataset, dataset)  # (N,D,D)
        Lam = utils.update_with_4_119(W, nu)  # (K,D,D)
        #
        a = torch.einsum('nde,kde->nk', Phi, Lam)  # (N,K)
        lm = utils.update_with_4_121(W, nu, m)  # (K,D)
        #
        b = torch.einsum('nd,kd->nk', dataset, lm)  # (N,K)
        c = utils.update_with_4_122(W, nu, m, beta).squeeze()  # (K)
        d = utils.update_with_4_120(W, nu).squeeze()  # (K)
        e = utils.update_with_4_62(alpha).squeeze()  # (K)
        # print("e ", e.size())

        f = torch.exp(-0.5 * a + b - 0.5 * c + 0.5 * d + e)  # (N,K)
        s = (1.0 / torch.sum(f, dim=1)).reshape(-1, 1)  # (N)
        return torch.einsum("nk,ne->nk", f, s)


class TestQsUpdater(unittest.TestCase):

    def test_0(self):
        N = 2
        D = 3
        dataset = torch.tensor([[1, 2, 3], [4, 5, 6]])
        p = torch.einsum('nd,ne->nde', dataset, dataset)
        self.assertTrue(p.size() == (N, D, D))
        answer = torch.tensor([[[1, 2, 3], [2, 4, 6], [3, 6, 9]], [[16, 20, 24], [20, 25, 30], [24, 30, 36]]])
        self.assertTrue(torch.all(answer == p))

    def test_1(self):
        N = 2
        D = 3
        K = 1
        a = torch.arange(N * D * D).reshape(N, D, D)
        b = torch.arange(K * D * D).reshape(K, D, D)
        c = torch.einsum('nde,kde->nk', a, b)
        self.assertTrue(c.size() == (N, K))

    def test_2(self):
        N = 2
        D = 3
        K = 1
        a = torch.arange(N * D).reshape(N, D)
        b = torch.arange(K * D).reshape(K, D)
        c = torch.einsum('nd,kd->nk', a, b)
        self.assertTrue(c.size() == (N, K))

    # W:(K,D,D), nu:(K,1), m:(K,D), beta:(K,1), alpha:(K,1)
    def test_3(self):
        N = 2
        D = 3
        K = 4
        dataset = torch.arange(1, 1 + N * D, dtype=float).reshape(N, D)
        t = torch.eye(D, dtype=float)
        s = [t for _ in range(K)]
        W = torch.stack(s, dim=0)
        nu = torch.arange(1, 1 + K, dtype=float).reshape(K, 1) / 100.0
        m = torch.arange(1, 1 + K * D, dtype=float).reshape(K, D)
        beta = torch.arange(1, 1 + K, dtype=float).reshape(K, 1)
        alpha = torch.arange(1, 1 + K, dtype=float).reshape(K, 1)

        updater = QsUpdater()
        a = updater.update(dataset, W, nu, m, beta, alpha)
        self.assertTrue(a.size() == (N, K))
        t = torch.sum(a, dim=1)
        self.assertAlmostEqual(1, t[0].item())
        self.assertAlmostEqual(1, t[1].item())


if __name__ == "__main__":
    unittest.main()
