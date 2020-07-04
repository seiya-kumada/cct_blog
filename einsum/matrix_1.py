#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np

if __name__ == "__main__":
    # 2つの行列を定義する。
    x = np.array([[1, 2, 3], [1, 2, 3]])
    y = np.array([[1, 3], [1, 3]])

    z = np.einsum("ni,nj->ij", x, y)
    assert(np.all(z == np.array([[2, 6], [4, 12], [6, 18]])))
