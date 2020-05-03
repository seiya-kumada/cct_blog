#!/usr/bin/env python
# -*- coding: utf-8 -*-
import coprime_number_generator as cng
import prime_factorize as pf
import numpy as np


def calculate(mi, ni, k, avs, bvs):
    gen = cng.Generator(mi, ni)
    values = gen.generate(k)

    n = 0
    for (a, b) in values:
        c = a + b
        r = pf.prime_factorize(a * b * c)
        r = list(set(r))
        d = np.prod(r)
        if c < d:
            pass
        else:  # c > d
            n += 1
            # print("(a,b,c,d)=({},{},{},{})".format(a, b, c, d))
            avs.append(a)
            bvs.append(b)

    print(">> {}/{}".format(n, len(values)))
    return len(values), n


if __name__ == "__main__":
    avs = []
    bvs = []
    s, m = calculate(2, 1, 7, avs, bvs)
    t, n = calculate(3, 1, 7, avs, bvs)
    print("{}/{}".format(m + n, s + t))

    with open("./cd_7.txt", "w") as fout:
        for (a, b) in zip(avs, bvs):
            fout.write("{} {}\n".format(a, b))
