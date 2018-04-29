#!/usr/bin/env python
# -*- coding: utf-8 -*-
# import pystan
# import numpy as np
import matplotlib.pyplot as plt


def define_model():
    return """
        data {
            int<lower=0> N; // the number of samples
            vector[N] x;
            vector[N] y;
        }
        parameters {
            real alpha;
            real beta;
            real<lower=0> sigma;
        }
        transformed parameters {
            int<lower=0> M; // the number of dimensions
            matrix[N, M] x_matrix;
            for (j in 0:N) {
                for (i in 0:M) {
                    matrix[i, j] = x[i]^j 
                }
            }
        }
        model {
            y ~ normal(alpha + beta * x, sigma);
        }
    """


def dataset_generator(path):
    for line in open(path):
        tokens = line.strip().split()
        if len(tokens) == 2:
            yield float(tokens[0]), float(tokens[1])


if __name__ == '__main__':
    # model = define_model()
    # print(model)
    xs = []
    ys = []
    for (x, y) in dataset_generator('./dataset.txt'):
        xs.append(x)
        ys.append(y)
    print(xs[0], ys[0])
    print(xs[-1], ys[-1])

    plt.scatter(xs, ys)
    plt.show()
