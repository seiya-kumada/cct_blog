#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
import math
import unittest


def update_with_4_119(W, nu):
    return nu * W


def update_with_4_120(W, nu):
    dim, _ = W.size()
    a = sum([torch.digamma(torch.tensor([(nu + 1.0 - d) * 0.5])).item() for d in range(dim)])
    b = torch.logdet(W).item()
    c = dim * math.log(2)
    return a + b + c


def update_with_4_121(W, nu, m):
    return nu * torch.matmul(W, m)


def update_with_4_122(W, nu, m, beta):
    dim, _ = W.size()
    return nu * m.dot(torch.matmul(W, m)) + dim / beta


def update_with_4_62(alpha, index_k):
    a = torch.sum(alpha)
    return torch.digamma(torch.tensor([alpha[index_k]])) - torch.digamma(torch.tensor([a]))


class TestUtils(unittest.TestCase):

    def test_update_with_4_119(self):
        nu = 5
        W = torch.eye(2)
        a = update_with_4_119(W, nu)
        self.assertTrue(torch.all(a == torch.tensor([[5, 0], [0, 5]])))

    def test_update_with_4_120(self):
        nu = 5
        W = torch.eye(2)
        a = update_with_4_120(W, nu)
        self.assertAlmostEqual(a, 3.01223533964696)

    def test_update_with_4_121(self):
        m = torch.tensor([2, 2.0])
        W = torch.eye(2)
        nu = 5
        a = update_with_4_121(W, nu, m)
        self.assertTrue(torch.all(a == torch.tensor([10.0, 10.0])))

    def test_update_with_4_122(self):
        m = torch.tensor([2, 2.0])
        W = torch.eye(2)
        nu = 5
        beta = 1
        a = update_with_4_122(W, nu, m, beta)
        self.assertTrue(42.0 == a)

    def test_update_with_4_62(self):
        alpha = torch.tensor([1.0, 1.0])
        a = update_with_4_62(alpha, 0)
        self.assertAlmostEqual(-1.0, a, delta=1.0e-5)


if __name__ == "__main__":
    unittest.main()
