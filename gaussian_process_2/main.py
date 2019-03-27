#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def exponential_cov(x, y, params):
    return params[0] * np.exp(-0.5 * params[1] * np.subtract.outer(x, y) ** 2)


def conditional(x_new, x, y, params):
    B = exponential_cov(x_new, x, params)
    C = exponential_cov(x, x, params)
    A = exponential_cov(x_new, x_new, params)
    mu = np.ligalg.inv(C).dot(B.T).T.dot(y)
    sigma = A - B.dot(np.linalg.inv(C).dot(B.T))
    return (mu.squeeze(), sigma.squeeze())


if __name__ == "__main__":
    theta = [1, 10]
    sigma_0 = exponential_cov(0, 0, theta)
    xpts = np.arange(-3, 3, step=0.01)
    plt.errorbar(xpts, np.zeros(len(xpts)), yerr=sigma_0, capsize=0)
    plt.ylim(-3, 3)
    plt.show()
