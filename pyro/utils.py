#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
# import math
import unittest


# test ok
def update_with_4_119(W, nu):
    # W:K,D,D
    # nu:K
    return torch.einsum("kde,k->kde", W, nu)


def f(nk, d):
    return torch.tensor((nk + 1 - d) / 2.0)


# test ok
def update_with_4_120(W, nu):
    # W:K,D,D
    # nu:K
    K, D, _ = W.size()
    p = torch.empty(K, D)
    for k in range(K):
        for d in range(D):
            p[k, d] = torch.digamma(f(nu[k].item(), 1 + d))
    x = torch.matmul(p, torch.ones(D))
    # y = torch.log(torch.abs(torch.det(W)))
    y = torch.logdet(W)
    z = x + D * torch.log(2.0 * torch.ones(K, dtype=float)) + y
    return z


# test ok
def update_with_4_121(W, nu, m):
    # W:K,D,D
    # nu:K
    # m:K,D
    a = update_with_4_119(W, nu)  # K,D,D
    b = torch.einsum("kde,ke->kd", a, m)  # K,D
    return b


# test ok
def update_with_4_122(W, nu, m, beta):
    # W:K,D,D
    # nu:K
    # m:K,D
    # beta:(K)
    M = torch.einsum("ki,kj->kij", m, m)  # (K,D,D)
    a = torch.einsum("kij,kij->k", M, W)  # (K,)
    _, D = m.size()
    return a * nu + D / beta


# test ok
def update_with_4_62(alpha):
    # alpha:K,1
    a = torch.sum(alpha)
    return (torch.digamma(alpha) - torch.digamma(a)).reshape(-1, 1)


class TestUtils(unittest.TestCase):

    def test_update_with_4_119(self):
        K = 3
        D = 2
        nu = torch.arange(K, dtype=float)
        self.assertTrue(nu.size() == (K,))
        W = torch.arange(K * D * D, dtype=float).reshape(K, D, D)
        a = update_with_4_119(W, nu)
        self.assertTrue(a.size() == (K, D, D))
        for k in range(K):
            self.assertTrue(torch.all(a[k] == nu[k] * W[k]))

    def test_update_with_4_120(self):
        K = 2
        D = 3
        t = torch.eye(D, dtype=float)
        s = [t for _ in range(K)]
        W = torch.stack(s, dim=0)  # torch.arange(1, 1 + K * D * D, dtype=float).reshape(K, D, D)
        nu = torch.arange(1, 1 + K, dtype=float) / 100.0
        a = update_with_4_120(W, nu)
        self.assertTrue(a.size() == (K,))

    def test_update_with_4_121(self):
        K = 3
        D = 2
        nu = torch.arange(K).reshape(K)
        W = torch.arange(K * D * D).reshape(K, D, D)
        m = torch.arange(K * D).reshape(K, D)
        a = update_with_4_121(W, nu, m)
        self.assertTrue(a.size() == (K, D))
        for k in range(K):
            self.assertTrue(torch.all(a[k] == torch.matmul(nu[k] * W[k], m[k])))

    def test_update_with_4_122(self):

        m = torch.tensor([[2, 1], [3, 2], [1, 5]])
        n = torch.einsum("ki,kj->kij", m, m)
        self.assertTrue(torch.all(n[0] == torch.tensor([[4, 2], [2, 1]])))
        self.assertTrue(torch.all(n[1] == torch.tensor([[9, 6], [6, 4]])))
        self.assertTrue(torch.all(n[2] == torch.tensor([[1, 5], [5, 25]])))

        K = 3
        D = 2
        a = torch.arange(K * D * D).reshape(K, D, D)
        b = torch.arange(K * D * D).reshape(K, D, D)
        c = torch.einsum("kij,kij->k", a, b)
        self.assertTrue(c.size() == (K,))

        d = torch.arange(K)
        e = torch.arange(K)
        f = d * e
        self.assertTrue(f.size() == (K,))

        K = 3
        D = 2
        nu = torch.arange(K)
        beta = torch.arange(K)
        W = torch.arange(K * D * D).reshape(K, D, D)
        m = torch.arange(K * D).reshape(K, D)
        a = update_with_4_122(W, nu, m, beta)
        self.assertTrue(a.size() == (K,))

    # def test_update_with_4_62(self):
    #     K = 2
    #     alpha = torch.ones(K, dtype=float)
    #     a = update_with_4_62(alpha)
    #     self.assertTrue(a.size() == (K, 1))


if __name__ == "__main__":
    unittest.main()
