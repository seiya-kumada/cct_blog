#!/usr/bin/env python
# -*- coding:utf-8 -*-
import blueqat as bq


if __name__ == "__main__":

    # 2量子ビットを使う回路
    c = bq.Circuit(2)

    # アダマールゲートを最初のビット(0)に適用
    x = c.h[0]

    # 2つのビット(0,1)にCNOTゲートを適用
    x = x.cx[0, 1]

    # 1000回測定する。
    y = x.m[:].run(shots=1000)

    print(y)
