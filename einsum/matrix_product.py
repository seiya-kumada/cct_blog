#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np

if __name__ == "__main__":
    # 2つの行列を定義する。
    A = np.array([[1, 2], [2, 1]])
    B = np.array([[0, 3], [3, 0]])

    # 積を計算する。matmulを使用した場合。
    C = np.matmul(A, B)
    assert(np.all(C == np.array([[6, 3], [3, 6]])))

    # einsumの場合。
    D = np.einsum("in,nj->ij", A, B)
    assert(np.all(C == D))
