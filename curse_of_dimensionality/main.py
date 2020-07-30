#!/usr/bin/env python
# -*- coding:utf-8 -*-
import math
import numpy as np
import matplotlib.pyplot as plt


class RadiusGauss:

    def __init__(self, D, sigma=1):
        S = 2 * math.pi ** (D / 2) / math.gamma(D / 2)
        self.sigma = sigma
        self.coeff = S / (2 * math.pi * sigma ** 2) ** (D / 2)
        self.D = D

    def __call__(self, x):
        return self.coeff * x ** (self.D - 1) * math.exp(-0.5 * x ** 2 / self.sigma ** 2)


if __name__ == "__main__":
    maxv = 1.5 * math.sqrt(100)
    xs = np.linspace(0, maxv, 100)
    rg_1 = RadiusGauss(1)
    rg_3 = RadiusGauss(3)
    rg_10 = RadiusGauss(10)
    rg_100 = RadiusGauss(100)
    ys_1 = [rg_1(x) for x in xs]
    ys_3 = [rg_3(x) for x in xs]
    ys_10 = [rg_10(x) for x in xs]
    ys_100 = [rg_100(x) for x in xs]
    plt.xlim(0, maxv)
    plt.plot(xs, ys_1, label="$D=1$")
    plt.plot(xs, ys_3, label="$D=3$")
    plt.plot(xs, ys_10, label="$D=10$")
    plt.plot(xs, ys_100, label="$D=100$")
    plt.xlabel("$r$")
    plt.ylabel("$q_D(r)$")
    plt.legend(loc="best")
    plt.savefig("./fig.jpg")
