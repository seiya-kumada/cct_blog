#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np

if __name__ == "__main__":
    # 2つの行列を定義する。
    x = np.array([[1, 2, 3], [1, 2, 3]])
    y = np.array([[1, 3], [1, 3]])

    z = np.einsum("ni,nj->ij", x, y)
    assert(np.all(z == np.array([[2, 6], [4, 12], [6, 18]])))


    x = np.arange(16).reshape(2, 2, 2, 2)
    y = np.arange(8).reshape(2, 2, 2)
    z = np.arange(16).reshape(2, 2, 2, 2)
    w = np.einsum("iabc,abc,abcj->ij", x, y, z)
    assert((2, 2) == w.shape)

