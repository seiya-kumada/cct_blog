#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np

if __name__ == "__main__":
    # 2つのベクトルを定義する。
    x = np.array([1, 2, 3])
    y = np.array([1, 3])

    z = np.einsum("i,j->ij", x, y)
    assert(np.all(z == np.array([[1, 3], [2, 6], [3, 9]])))
