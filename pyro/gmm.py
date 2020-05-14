#!/usr/bin/env python
# -*- coding:utf-8 -*-
import parameters as pa
import qs_updater as qs
import qpi_updater as qp
import qmusigma_updater as qm
import torch
import torch.distributions as D
import matplotlib.pyplot as plt
import numpy as np
import random


DIM = 2
K = 3
NU = 1000
MAX_ITER = 1
OBS_NUM = 30
SEED = 1
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)


def display_graph(dataset):
    xs = []
    ys = []
    for (x, y) in dataset.numpy():
        xs.append(x)
        ys.append(y)
    plt.scatter(xs, ys, marker='.')
    plt.show()


def make_dataset(obs_num, dim):
    loc_0 = torch.tensor([-10.0, 0])
    cov_0 = torch.eye(dim) * 2.0
    dis_0 = D.MultivariateNormal(loc=loc_0, covariance_matrix=cov_0)

    loc_1 = torch.tensor([10.0, 0])
    cov_1 = torch.eye(dim) * 2.0
    dis_1 = D.MultivariateNormal(loc=loc_1, covariance_matrix=cov_1)

    loc_2 = torch.tensor([0, 10.0])
    cov_2 = torch.eye(dim) * 2.0
    dis_2 = D.MultivariateNormal(loc=loc_2, covariance_matrix=cov_2)

    values = []
    for _ in range(OBS_NUM // K):
        a = dis_0.sample()
        b = dis_1.sample()
        c = dis_2.sample()
        values.append(a)
        values.append(b)
        values.append(c)
    return torch.stack(values, dim=0)


if __name__ == "__main__":
    try:
        hyperparams = pa.HyperParameters(dim=DIM, k=K, nu=NU)
        params = pa.Parameters(dim=DIM, k=K)
        qs_updater = qs.QsUpdater(params.eta)
        qp_updater = qp.QpiUpdater()
        qm_updater = qm.QmusigmaUpdater()
        dataset = make_dataset(OBS_NUM, DIM)

        # display_graph(dataset)

        for _ in range(MAX_ITER):
            qs_updater.update(
                hyperparams.W, hyperparams.nu, hyperparams.m, hyperparams.beta, hyperparams.alpha, dataset)
        #     for _ in range(K):
        #         qp_updater.update()
        #     qm_updater.update()
    except Exception as e:
        print("Exception: {}".format(e))
