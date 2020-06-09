#!/usr/bin/env python
# -*- coding:utf-8 -*-
import parameters as pa
import qs_updater as qs
import qpi_updater as qp
import qmu_updater as qm
import qlambda_updater as ql
import torch
import dataset as ds
# import matplotlib.pyplot as plt
import numpy as np
import gauss
import wishart
# import random


DIM = 4
K = 4
NU = DIM * torch.ones(K)
MAX_ITER = 100
OBS_NUM = 100
# SEED = 1
EPSILON = 1.0e-5
TRIAL_NUM = 1

# RED = np.array([1.0, 0.0, 0.0])
# GREEN = np.array([0.0, 1.0, 0.0])
# BLUE = np.array([0.0, 0.0, 1.0])


# torch.manual_seed(SEED)
# random.seed(SEED)
# np.random.seed(SEED)


# (N,D)
def find_minmax(dataset):
    mins = torch.min(dataset, dim=0)
    maxs = torch.max(dataset, dim=0)
    return mins.values, maxs.values


class Predictor:

    def __init__(self, ql_updater, qm_updater, qp_updater):
        self.wishs = [wishart.Wishart(ql_updater.nu.numpy()[k], ql_updater.W.numpy()[k]) for k in range(K)]
        self.ql_updater = ql_updater
        self.qm_updater = qm_updater
        self.qp_updater = qp_updater

    def predict(self, x):
        group = []
        eta = qs.QsUpdater.calculate_eta(
            x.reshape(1, -1),
            ql_updater.W,
            ql_updater.nu,
            qm_updater.m,
            qm_updater.beta,
            qp_updater.alpha
        )[0]

        for _ in range(TRIAL_NUM):
            ys = []
            for k in range(K):
                Lambda = self.wishs[k].sample().astype(np.float32)
                g_mu = gauss.Gauss(qm_updater.m[k], qm_updater.beta[k] * Lambda)
                mu = g_mu.sample()
                g_x = gauss.Gauss(mu, torch.tensor(Lambda))
                y = eta[k] * g_x.probs(x)
                ys.append(y)
            ys /= np.sum(ys)
            group.append(ys)
        return np.array(group)


# eta:(N,K), dataset:(N,D)
# def save_results(eta, dataset):
#     # red = np.array([1, 0, 0])
#     # green = np.array([0, 1, 0])
#     # blue = np.array([0, 0, 1])
#
#     colors = []
#     for indices in eta:
#         c = RED * indices[0].numpy() + GREEN * indices[1].numpy() + BLUE * indices[2].numpy()
#         colors.append(c)
#
#     plt.figure(figsize=(5, 5))
#     plt.axes().set_aspect("equal")
#     plt.scatter(dataset[:, 0], dataset[:, 1], marker='.', c=colors)
#
#     plt.xlim(X_MIN, X_MAX)
#     plt.ylim(Y_MIN, Y_MAX)
#     plt.savefig('./results.jpg')


# def predict(ql_updater, qm_updater, qp_updater):
#     pred = Predictor(ql_updater, qm_updater, qp_updater)
#     h = 0.025
#     colors = []
#     xs, ys = np.meshgrid(np.arange(X_MIN, X_MAX, h).astype(np.float32),  np.arange(Y_MIN, Y_MAX, h).astype(np.float32))
#     for x, y in zip(xs.ravel(), ys.ravel()):
#         z = pred.predict(torch.tensor([x, y]))
#         az = np.mean(z, axis=0)
#         c = RED * az[0] + GREEN * az[1] + BLUE * az[2]
#         colors.append(c)
#     return xs, ys, colors


# def save_all_results(eta, dataset, xs, ys, pcolors):
#
#     plt.figure(figsize=(5, 5))
#     plt.axes().set_aspect("equal")
#
#     colors = []
#     for indices in eta:
#         c = RED * indices[0].numpy() + GREEN * indices[1].numpy() + BLUE * indices[2].numpy()
#         colors.append(c)
#
#     plt.scatter(dataset[:, 0], dataset[:, 1], marker='.', c=colors)
#     plt.scatter(xs.ravel(), ys.ravel(), marker=".", c=pcolors, alpha=0.1)
#     plt.xlim(X_MIN, X_MAX)
#     plt.ylim(Y_MIN, Y_MAX)
#     plt.savefig('./predict.jpg')


def make_initial_positions(mins, maxs):
    cs = []
    for (mi, ma) in zip(mins, maxs):
        c = np.random.uniform(mi.item(), ma.item(), K)  # (3,)
        cs.append(c)
    cs = np.array(cs).transpose()
    return cs  # (K,D)


def print_mu(mu, std, means):
    for m in mu * std + mean:
        print(m.tolist())


if __name__ == "__main__":
    try:
        hyper_params = pa.HyperParameters(dim=DIM, k=K, nu=NU)
        qs_updater = qs.QsUpdater()
        qp_updater = qp.QpiUpdater(hyper_params)
        qm_updater = qm.QmuUpdater(hyper_params)
        ql_updater = ql.QlambdaUpdater(hyper_params)
        dataset = ds.make_d_dataset(OBS_NUM, DIM, K)
        std, mean = torch.std_mean(dataset, dim=0)  # (N,D)
        dataset = (dataset - mean) / std
        mins, maxs = find_minmax(dataset)
        cs = make_initial_positions(mins, maxs)

        is_ok = False
        # initialize mu
        qm_updater.m = torch.tensor(cs).float()

        print_mu(qm_updater.m, std, mean)

        prev_m = qm_updater.m.clone()
        for i in range(MAX_ITER):
            qs_updater.update(
                dataset,
                ql_updater.W,
                ql_updater.nu,
                qm_updater.m,
                qm_updater.beta,
                qp_updater.alpha)
            ql_updater.update(dataset, qs_updater.eta, qm_updater.beta, qm_updater.m)
            qm_updater.update(dataset, qs_updater.eta)
            qp_updater.update(dataset, qs_updater.eta)
            diff_m = torch.max(torch.abs(qm_updater.m - prev_m))
            # print("> diff {} {} {}".format(diff_m, qm_updater.m, prev_m))
            if diff_m < EPSILON:
                # print("> diff is {} at {}".format(diff_m, i))
                is_ok = True
                break
            prev_m = qm_updater.m.clone()

        if is_ok:
            print("> SUCCESS!")
        else:
            print("> ERROR!")

        print_mu(qm_updater.m, std, mean)

        # xs, ys, colors = predict(ql_updater, qm_updater, qp_updater)
        # save_all_results(qs_updater.eta, dataset, xs, ys, colors)
    except Exception as e:
        print("Exception: {}".format(e))
