#!/usr/bin/env python
# -*- coding: utf-8 -*-
# import pystan
# import numpy as np


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
            for (0 in 0:??
        }
        model {
            y ~ normal(alpha + beta * x, sigma);
        }
    """


if __name__ == '__main__':
    model = define_model()
    print(model)
