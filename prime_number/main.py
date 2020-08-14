#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

THETAS = [
    14.1347,
    21.0220,
    25.0109,
    30.4249,
    32.9351,
    37.5862,
    40.9187,
    43.3271,
    48.0052,
    49.7738,
    52.9703,
    56.4462,
    59.3470,
    60.8318,
    65.1125,
]

MIN_X = 1.5
MAX_X = 15
N = 1000

if __name__ == "__main__":

    xs = np.linspace(MIN_X, MAX_X, N)

    for i in range(len(THETAS)):
        ys = [0] * N
        for t in THETAS[:i]:
            ys -= np.cos(t * np.log(xs)) / np.sqrt(xs)

        max_y = np.max(ys)
        min_y = np.min(ys)
        plt.rcParams["font.size"] = 12
        plt.figure(figsize=(10, 4))
        plt.plot(xs, ys, label="till {:02}".format(i))
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        plt.xticks([2, 3, 5, 7, 8, 9, 11, 13], ["2", "3", "5", "7", "(8)", "(9)", "11", "13"])
        plt.vlines([2, 3, 5, 7, 11, 13], ymin=min_y, ymax=max_y, linestyles='dotted')
        plt.legend(loc="upper right")
        plt.savefig("./primes_{:02}.jpg".format(i))
        plt.clf()
