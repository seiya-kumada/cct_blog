#!/usr/bin/env python
# -*- using:utf-8 -*-
import math
from blueqat import Circuit
# https://qiita.com/KeiichiroHiga/items/7fae38f8c2004dfcd246


def qft(x):
    ll = len(x)
    lb = format(ll - 1, 'b')
    b = len(lb)

    # 状態|0>に演算子Fを作用させる。
    F = Circuit()  # 初期状態は|0>
    for i in range(b):
        F.h[i]
        for j in range(1, b - i):
            F.cphase(math.pi / 2**(j + 1))[i + j, i]

    w = []
    for j in range(ll):
        c = Circuit()
        b = format(j, 'b')
        bl = len(b)

        # Fが作用する状態|j>を作る。
        for a in range(bl):
            if b[bl - a - 1] == '1':
                c = c.x[len(lb) - 1 - a]

        # F|0>,F|1>,...を計算する。
        # 今作った状態に結合すれば良い。
        d = c + F
        e = d.run()  # w_k,k=0,1,...
        w.append(e)

    y = []
    for k in range(ll):
        y_k = 0
        for j in range(ll):
            y_k += w[j][k] * x[j]
        y.append(y_k)
    return y


def qft_with_2qbit_00():
    # |j0j1>=|00>
    c = Circuit(2)

    # j0ビットにアダマールゲートをかける。
    c = c.h[0]

    # j1ビットを制御ビットとしてj0ビットにR1をかける。
    c = c.cphase(math.pi / 2)[1, 0]

    # j1ビットにアダマールゲートをかける。
    c = c.h[1]

    # 実行。
    r = c.run()

    print("> 2qbit00")
    for i, v in enumerate(r):
        b = "{:02b}".format(i)
        print("[{}]{}".format(b, v))


def qft_with_2qbit_11():
    # |j0j1>=|00>
    c = Circuit(2)

    # 両ビットにXゲートをかける。Xゲートはビットを反転させる計算である。状態は|11>になる。
    c = c.x[0, 1]

    # j0ビットにアダマールゲートをかける。
    c = c.h[0]

    # j1ビットを制御ビットとしてj0ビットにR1をかける。
    c = c.cphase(math.pi / 2)[1, 0]

    # j1ビットにアダマールゲートをかける。
    c = c.h[1]

    # 実行。
    r = c.run()

    print("> 2qbit11")
    for i, v in enumerate(r):
        b = "{:02b}".format(i)
        print("[{}]{}".format(b, v))


def qft_with_3qbit_110():
    # |j0j1j2> = |000>
    c = Circuit(3)

    # |j0j1j2> = |110>
    c = c.x[0, 1]

    # j0ビットにアダマールゲートをかける。
    c = c.h[0]

    # j1ビットを制御ビットにしてj0ビットにR1をかける。
    c = c.cphase(math.pi / 2)[1, 0]

    # j2ビットを制御ビットにしてj0ビットにR2をかける。
    c = c.cphase(math.pi / 4)[2, 0]

    # j2ビットを制御ビットにしてj1ビットにR1をかける。
    c.h[1].cphase(math.pi / 2)[2, 1]

    # j2ビットにアダマールゲートをかける。
    c.h[2]

    r = c.run()
    r = list(2 * math.sqrt(2) * r)
    print("> 3qbit")
    for i, v in enumerate(r):
        b = "{:03b}".format(i)
        print("[{}]{}".format(b, v))


if __name__ == "__main__":
    y = qft([1, 1, 1, 1])
    print(y)
