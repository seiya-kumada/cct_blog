#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np

if __name__ == "__main__":
    # 2つの行列を定義する。
    x = np.array([[1, 2], [2, 1]])
    y = np.array([[0, 3], [3, 0]])

    # 積を計算する。matmulを使用した場合。
    z = np.matmul(x, y)
    assert(np.all(z == np.array([[6, 3], [3, 6]])))

    # einsumの場合。
    u = np.einsum("in,nj->ij", x, y)
    assert(np.all(u == z))
