#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
import math
import unittest


# test ok
def update_with_4_119(W, nu):
    # W:K,D,D
    # nu:K,1
    return torch.einsum("kde,kf->kde", W, nu)


def update_with_4_120(W, nu):
    # W:K,D,D
    # nu:K,1

    _, D, _ = W.size()
    a = sum([torch.digamma(torch.tensor([(nu + 1.0 - d) * 0.5])).item() for d in range(D)])
    b = torch.logdet(W).item()
    c = D * math.log(2)
    return a + b + c


# test ok
def update_with_4_121(W, nu, m):
    # W:K,D,D
    # nu:K,1
    # m:K,D
    a = update_with_4_119(W, nu)  # K,D,D
    b = torch.einsum("kde,ke->kd", a, m)  # K,D
    return b


# test ok
def update_with_4_122(W, nu, m, beta):
    # W:K,D,D
    # nu:K,1
    # m:K,D
    # beta:(K,1)
    M = torch.einsum("ki,kj->kij", m, m)  # (K,D,D)
    a = torch.einsum("kij,kij->k", M, W).reshape(-1, 1)  # (K,1)
    _, D = m.size()
    return a * nu + D / beta


def update_with_4_62(alpha, index_k):
    a = torch.sum(alpha)
    return torch.digamma(torch.tensor([alpha[index_k]])) - torch.digamma(torch.tensor([a]))


class TestUtils(unittest.TestCase):

    def test_update_with_4_119(self):
        K = 3
        D = 2
        nu = torch.arange(K).reshape(K, 1)
        self.assertTrue(nu.size() == (K, 1))
        W = torch.arange(12).reshape(K, D, D)
        a = update_with_4_119(W, nu)
        self.assertTrue(a.size() == (K, D, D))
        for k in range(K):
            self.assertTrue(torch.all(a[k] == nu[k] * W[k]))

    def test_update_with_4_120(self):
        nu = 5
        W = torch.eye(2)
        a = update_with_4_120(W, nu)
        self.assertAlmostEqual(a, 3.01223533964696)

    def test_update_with_4_121(self):
        K = 3
        D = 2
        nu = torch.arange(K).reshape(K, 1)
        W = torch.arange(K * D * D).reshape(K, D, D)
        m = torch.arange(K * D).reshape(K, D)
        a = update_with_4_121(W, nu, m)
        self.assertTrue(a.size() == (K, D))
        for k in range(K):
            self.assertTrue(torch.all(a[k] == torch.matmul(nu[k] * W[k], m[k])))

    def test_update_with_4_122(self):
        K = 3
        D = 2
        a = torch.arange(K * D * D).reshape(K, D, D)
        b = torch.arange(K * D * D).reshape(K, D, D)
        c = torch.einsum("kij,kij->k", a, b).reshape(-1, 1)
        self.assertTrue(c.size() == (K, 1))

        d = torch.arange(K * 1).reshape(K, 1)
        e = torch.arange(K * 1).reshape(K, 1)
        f = d * e
        self.assertTrue(f.size() == (K, 1))

        K = 3
        D = 2
        nu = torch.arange(K).reshape(K, 1)
        beta = torch.arange(K).reshape(K, 1)
        W = torch.arange(K * D * D).reshape(K, D, D)
        m = torch.arange(K * D).reshape(K, D)
        a = update_with_4_122(W, nu, m, beta)
        self.assertTrue(a.size() == (K, 1))

    def test_update_with_4_62(self):
        alpha = torch.tensor([1.0, 1.0])
        a = update_with_4_62(alpha, 0)
        self.assertAlmostEqual(-1.0, a, delta=1.0e-5)


if __name__ == "__main__":
    unittest.main()
