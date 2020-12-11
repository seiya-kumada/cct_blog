#!/usr/bin/env python
# -*- using:utf-8 -*-
import math
from blueqat import Circuit


# 配列が入力[x_0,...,x_N-1}
def qft(init):
    a = len(init)  # 4
    lb = format(a - 1, 'b')  # 11
    b = len(lb)  # 2

    c_qft = Circuit()
    for i in range(b):  # i=0,1
        c_qft.h[i]
        for j in range(1, b - i):  # when i=0, j=1, when i=1
            c_qft.phase(math.pi / 2**(j + 1))[i + j, i]


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
    qft([1, 2, 3, 4])
