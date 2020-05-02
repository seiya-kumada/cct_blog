#!/usr/bin/env python
# -*- coding: utf-8 -*-
import coprime_number_generator as cng
import prime_factorize as pf
import numpy as np


def calculate(mi, ni, k):
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
            print("(a,b,c,d)=({},{},{},{})".format(a, b, c, d))

    print(">> {}/{}".format(n, len(values)))
    return len(values), n


if __name__ == "__main__":
    s, m = calculate(2, 1, 10)
    t, n = calculate(3, 1, 10)
    print("{}/{}".format(m + n, s + t))
