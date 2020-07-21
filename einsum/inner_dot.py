#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np

if __name__ == "__main__":
    # 2つのベクトルを定義する。
    x = np.array([1, 2, 3])
    y = np.array([3, 2, 1])

    # 内積を計算する。dotを使用した場合。
    z = x.dot(y)
    assert(z == 10)

    # einsumの場合。
    u = np.einsum("i,i->", x, y)
    assert(u == z)
