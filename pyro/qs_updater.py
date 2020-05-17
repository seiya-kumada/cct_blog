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
        print("a ", a)
        lm = utils.update_with_4_121(W, nu, m)  # (K,D)
        #
        b = torch.einsum('nd,kd->nk', dataset, lm)  # (N,K)
        print("b ", b)

        c = utils.update_with_4_122(W, nu, m, beta).squeeze()  # (K)
        print("c ", c)
        d = utils.update_with_4_120(W, nu).squeeze()  # (K)
        print("d ", d)
        e = utils.update_with_4_62(alpha).squeeze()  # (K)
        # print("e ", e.size())

        f = torch.exp(-0.5 * a + b - 0.5 * c + 0.5 * d + e)  # (N,K)
        # print("f ", f.size())
        s = (1.0 / torch.sum(f, dim=1)).reshape(-1, 1)  # (N)
        # print("s ", s.size())
        return torch.einsum("nk,ne->nk", f, s)

    # def update(self, dataset, W, nu, m, beta, alpha):
    #     N, _ = dataset.size()
    #     r = torch.empty(N, self.K)
    #     for n in range(N):
    #         x = dataset[n]
    #         for k in range(self.K):
    #             Wk = W[k]
    #             nk = nu[k]
    #             Lambda = utils.update_with_4_119(Wk, nk)
    #             a = torch.dot(x, torch.matmul(Lambda, x))
    #             mk = m[k]
    #             Lambda_mu = utils.update_with_4_121(Wk, nk, mk)
    #             b = x.dot(Lambda_mu)
    #             bk = beta[k]
    #             c = utils.update_with_122(Wk, nk, mk, bk)
    #             d = utils.update_with_120(Wk, nk)
    #             alpha_k = alpha[k]
    #             e = utils.update_with_4_62(alpha_k, k)
    #             f = -0.5 * a + b - 0.5 * c + 0.5 * d + e
    #             r[n, k] = torch.exp(f)

    #     factors = torch.sum(r, dim=1)
    #     for n in range(N):
    #         r[n] = r[n] / factors[n]
    #     self.eta = r


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
        print("N,D,K ", N, D, K)
        dataset = torch.arange(1, 1 + N * D, dtype=float).reshape(N, D)
        W = torch.arange(1, 1 + K * D * D, dtype=float).reshape(K, D, D)
        nu = torch.arange(1, 1 + K, dtype=float).reshape(K, 1)
        m = torch.arange(1, 1 + K * D, dtype=float).reshape(K, D)
        beta = torch.arange(1, 1 + K, dtype=float).reshape(K, 1)
        alpha = torch.arange(1, 1 + K, dtype=float).reshape(K, 1)

        updater = QsUpdater()
        a = updater.update(dataset, W, nu, m, beta, alpha)
        self.assertTrue(a.size() == (N, K))
        print(torch.sum(a, dim=1))


if __name__ == "__main__":
    unittest.main()
