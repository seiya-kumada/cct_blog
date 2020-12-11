#!/usr/bin/env python
# -*- using:utf-8 -*-
import math
from blueqat import Circuit


def sample_blueqat_2_cphase():
    c = Circuit().x[0, 1].h[0].cphase(math.pi / 2)[1, 0].h[1]
    r = c.run()
    print("> ", r)
    print(">> ", pow(abs(r), 2))


def sample_blueqat_2_crz():
    c = Circuit().x[0, 1].h[0].crz(math.pi / 2)[1, 0].h[1]
    r = c.run()
    print("> ", r)
    print(">> ", pow(abs(r), 2))


def sample_blueqat_2_cu1():
    c = Circuit().x[0, 1].h[0].cu1(math.pi / 2)[1, 0].h[1]
    r = c.run()
    print("> ", r)
    print(">> ", pow(abs(r), 2))


def sample_blueqat_3():
    c = Circuit().x[0, 1].h[0].cphase(math.pi / 2)[1, 0].cphase(math.pi / 4)[2, 0]  # 0番目
    c.h[1].cphase(math.pi / 2)[2, 1]  # 1番目
    c.h[2]  # 2番目
    r = c.run()
    print(r)


if __name__ == "__main__":
    sample_blueqat_2_cphase()
    sample_blueqat_2_crz()
    sample_blueqat_2_cu1()
    # sample_blueqat_3()
