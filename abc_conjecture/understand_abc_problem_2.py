#!/usr/bin/env python
# -*- coding: utf-8 -*-
# import coprime_number_generator as cng
import prime_factorize as pf
import numpy as np
import math


if __name__ == "__main__":
    total = 8
    n = 0
    epsilon = 0.0
    for i in range(total):
        j = i + 1
        c = int(math.pow(3,  math.pow(2, j)))
        b = c - 1
        a = 1
        c = a + b
        r = pf.prime_factorize(a * b * c)
        r = list(set(r))
        d = np.prod(r)
        e = pow(d, 1 + epsilon)
        if c < e:
            n += 1
            print("c < e (a,b,c,e)=({}: {},{},{},{})".format(j, a, b, c, e))
        else:
            print("c > e (a,b,c,e)=({}: {},{},{},{})".format(j, a, b, c, e))

    print(">> {}/{}".format(n, total))
