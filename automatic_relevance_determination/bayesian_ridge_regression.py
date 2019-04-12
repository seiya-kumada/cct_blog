#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sklearn import linear_model


if __name__ == "__main__":
    X = [
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0]]
    Y = [
        0.0,
        1.0,
        2.0,
        3.0]

    reg = linear_model.BayesianRidge()
    reg.fit(X, Y)

    y = reg.predict([[1, 0]])
    print(y)
    print(reg.coef_)
