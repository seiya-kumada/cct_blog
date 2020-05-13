#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
import math


def update_4_119(W, nu):
    return nu * W


def update_4_120(W, nu):
    dim, _ = W.size()
    a = sum([torch.digamma(torch.tensor([(nu + 1.0 - d) * 0.5])).item() for d in range(dim)])
    b = torch.logdet(W).item()
    c = dim * math.log(2)
    return a + b + c
