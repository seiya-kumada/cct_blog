#!/usr/bin/env python
# -*- coding:utf-8 -*-
import hyperparameters as hp


if __name__ == "__main__":
    try:
        hyperparams = hp.HyperParameters(dim=1000, k=3, nu=1000)
    except Exception as e:
        print(e)
