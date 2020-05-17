#!/usr/bin/env python
# -*- coding:utf-8 -*-
import utils
import unittest
import torch


class QsUpdater:

    def __init__(self, N, K):
        self.K = K
        self.eta = torch.empty(N, K)

    # W:(K,D,D), nu:(K,1), m:(K,D)
    def update(self, dataset, W, nu, m, beta, alpha):
        Phi = torch.einsum('nd,ne->nde', dataset, dataset)  # (N,D,D)
        Lam = utils.update_with_4_119(W, nu)  # (K,D,D)
        a = torch.einsum('nde,kde->nk', Phi, Lam)  # (N,K)

        lm = utils.update_with_4_121(W, nu, m)  # (K,D)
        b = torch.einsum('nd,kd->nk', dataset, lm)  # (N,K)

        c = utils.update_with_122(W, nu, m, beta)  # (K,1)


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

        # dataset = torch.tensor([[2.0, 1.0], [3.0, 2.0]])
        # K = 3
        # eta = torch.empty(2, K)
        # updater = QsUpdater(eta, K=K)
        # W = torch.empty(K, 2, 2)
        # nu = torch.empty(K, 2)
        # m = torch.empty(K, 2)
        # beta = torch.emtpy(K)
        # alpha = torch.empty(K)
        # updater.update(W, nu, m, beta, alpha, dataset)


if __name__ == "__main__":
    unittest.main()
