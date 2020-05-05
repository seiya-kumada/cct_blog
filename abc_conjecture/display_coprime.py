#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

N = 3
PATH = "./coprime_{}.txt".format(N)
PATH_CD = "./cd_{}.txt".format(N)


if __name__ == "__main__":

    xs = []
    ys = []
    for line in open(PATH):
        tokens = line.split()
        xs.append(int(tokens[0]))
        ys.append(int(tokens[1]))

    cdxs = []
    cdys = []
    for line in open(PATH_CD):
        tokens = line.split()
        cdxs.append(int(tokens[0]))
        cdys.append(int(tokens[1]))

    plt.rcParams["font.size"] = 11
    plt.axes().set_aspect('equal')
    plt.xlabel("$a$")
    plt.ylabel("$b$")
    plt.scatter(xs, ys, marker=".")
    plt.scatter(cdxs, cdys, marker=".")
    plt.savefig("./dist_{}.jpg".format(N))
